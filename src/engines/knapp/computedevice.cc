#include <nba/core/intrinsic.hh>
#include <nba/framework/logging.hh>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/hosttypes.hh>
#include <nba/engines/knapp/hostutils.hh>
#include <nba/engines/knapp/computedevice.hh>
#include <nba/engines/knapp/computecontext.hh>
#include <nba/engines/knapp/ctrl.pb.h>
#include <nba/engines/knapp/rma.hh>
#include <rte_memory.h>
#include <scif.h>

using namespace std;
using namespace nba;
using namespace nba::knapp;

KnappComputeDevice::KnappComputeDevice(
        unsigned node_id, unsigned device_id, size_t num_contexts
) : ComputeDevice(node_id, device_id, num_contexts)
{
    int rc;
    type_name = "knapp.phi";
    assert(num_contexts > 0);
    RTE_LOG(DEBUG, COPROC, "KnappComputeDevice: # contexts: %lu\n", num_contexts);

    ctrl_epd = scif_open();
    struct scif_portID peer;
    peer.node = knapp::remote_scif_nodes[0];
    peer.port = KNAPP_CTRL_PORT;
    rc = scif_connect(ctrl_epd, &peer);
    assert(0 < rc);

    /* Check availability. */
    {
        knapp::CtrlRequest request;
        knapp::CtrlResponse response;
        request.set_type(knapp::CtrlRequest::PING);
        request.mutable_text()->set_msg("hello");
        ctrl_invoke(ctrl_epd, request, response);
        assert(knapp::CtrlResponse::SUCCESS == response.reply());
        assert("hello" == response.text().msg());
    }
    RTE_LOG(DEBUG, COPROC, "KnappComputeDevice: connected.\n");

    for (unsigned i = 0; i < num_contexts; i++) {
        KnappComputeContext *ctx = nullptr;
        NEW(node_id, ctx, KnappComputeContext, i, this);
        _ready_contexts.push_back(ctx);
        contexts.push_back((ComputeContext *) ctx);
    }
}

KnappComputeDevice::~KnappComputeDevice()
{
    for (auto ctx : _ready_contexts) {
        delete ctx;
    }
    for (auto ctx : _active_contexts) {
        delete ctx;
    }
    scif_close(ctrl_epd);
}

static uint32_t _knapp_cctx_buffer_count = 0;

uint32_t KnappComputeDevice::find_new_buffer_id()
{
    /* We assume that it is rare to free buffers
     * and allocate others in datap-palen apps. */
    assert(_knapp_cctx_buffer_count < KNAPP_GLOBAL_MAX_RMABUFFERS);
    return ++ _knapp_cctx_buffer_count;
}

int KnappComputeDevice::get_spec(struct compute_device_spec *spec)
{
    // FIXME: generalize for different Phi processor models.
    spec->max_threads = 60 * 4;
    spec->max_workgroups = 60;
    spec->max_concurrent_kernels = true;
    spec->global_memory_size = 8ll * 1024 * 1024 * 1024;
    return 0;
}

int KnappComputeDevice::get_utilization(struct compute_device_util *util)
{
    // FIXME: implement MIC status checks
    size_t free = 8ll * 1024 * 1024 * 1024, total = 8ll * 1024 * 1024 * 1024;
    util->used_memory_bytes = total - free;
    util->utilization = (float) free / total;
    return 0;
}

ComputeContext *KnappComputeDevice::_get_available_context()
{
    _ready_cond.lock();
    KnappComputeContext *cctx = _ready_contexts.front();
    assert(cctx != NULL);
    _ready_contexts.pop_front();
    _active_contexts.push_back(cctx);
    _ready_cond.unlock();
    return (ComputeContext *) cctx;
}

void KnappComputeDevice::_return_context(ComputeContext *cctx)
{
    /* This method is called inside Knapp's own thread. */
    assert(cctx != NULL);
    /* We do linear search here, it would not be a big overhead since
     * the number of contexts are small (less than 16 for Knapp). */
    _ready_cond.lock();
    assert(_ready_contexts.size() < num_contexts);
    for (auto it = _active_contexts.begin(); it != _active_contexts.end(); it++) {
        if (cctx == *it) {
            _active_contexts.erase(it);
            _ready_contexts.push_back(dynamic_cast<KnappComputeContext*>(cctx));
            break;
        }
    }
    _ready_cond.unlock();
}

host_mem_t KnappComputeDevice::alloc_host_buffer(size_t size, int flags)
{
    size_t aligned_size = ALIGN_CEIL(size, PAGE_SIZE);
    CtrlRequest request;
    CtrlResponse response;
    uint32_t buffer_id = find_new_buffer_id();
    const auto &search = buffer_registry.find(buffer_id);
    assert(search == buffer_registry.end());

    RMABuffer *buf = new RMABuffer(ctrl_epd, aligned_size, 0);
    request.set_type(CtrlRequest::CREATE_RMABUFFER);
    CtrlRequest::RMABufferParam *rma_param = request.mutable_rma();
    rma_param->set_vdev_handle((uintptr_t) 0);  // global
    rma_param->set_buffer_id(buffer_id);
    rma_param->set_size(aligned_size);
    rma_param->set_local_ra((uint64_t) buf->ra());
    ctrl_invoke(ctrl_epd, request, response);
    assert(CtrlResponse::SUCCESS == response.reply());
    buf->set_peer_ra(response.resource().peer_ra());
    buf->set_peer_va(response.resource().peer_va());
    buffer_registry.insert({ buffer_id, buf });
    host_mem_t m;
    m.buffer_id = buffer_id;
    return m;
}

dev_mem_t KnappComputeDevice::alloc_device_buffer(size_t size, int flags, host_mem_t &assoc_host_buf)
{
    size_t aligned_size = ALIGN_CEIL(size, PAGE_SIZE);
    uint32_t buffer_id = assoc_host_buf.buffer_id;
    const auto &search = buffer_registry.find(buffer_id);
    if (search == buffer_registry.end()) {
        CtrlRequest request;
        CtrlResponse response;
        buffer_id = find_new_buffer_id();
        RMABuffer *buf = new RMABuffer(ctrl_epd, aligned_size, 0);
        request.set_type(CtrlRequest::CREATE_RMABUFFER);
        CtrlRequest::RMABufferParam *rma_param = request.mutable_rma();
        rma_param->set_vdev_handle((uintptr_t) 0);  // global
        rma_param->set_buffer_id(buffer_id);
        rma_param->set_size(aligned_size);
        rma_param->set_local_ra((uint64_t) buf->ra());
        ctrl_invoke(ctrl_epd, request, response);
        assert(CtrlResponse::SUCCESS == response.reply());
        buf->set_peer_ra(response.resource().peer_ra());
        buf->set_peer_va(response.resource().peer_va());
        buffer_registry.insert({ buffer_id, buf });
    }
    dev_mem_t m;
    m.buffer_id = buffer_id;
    return m;
}

void KnappComputeDevice::free_host_buffer(host_mem_t m)
{
    return;
}

void KnappComputeDevice::free_device_buffer(dev_mem_t m)
{
    const auto &search = buffer_registry.find(m.buffer_id);
    if (search == buffer_registry.end()) {
        return;
    } else {
        CtrlRequest request;
        CtrlResponse response;
        request.set_type(CtrlRequest::DESTROY_RMABUFFER);
        request.mutable_rma_ref()->set_vdev_handle((uintptr_t) 0); // global
        request.mutable_rma_ref()->set_buffer_id(m.buffer_id);
        ctrl_invoke(ctrl_epd, request, response);
        assert(CtrlResponse::SUCCESS == response.reply());
        buffer_registry.erase(m.buffer_id);
        delete (*search).second;
    }
}

void *KnappComputeDevice::unwrap_host_buffer(const host_mem_t m)
{
    const auto &search = buffer_registry.find(m.buffer_id);
    assert(search != buffer_registry.end());
    return (void *) (*search).second->va();
}

void *KnappComputeDevice::unwrap_device_buffer(const dev_mem_t m)
{
    const auto &search = buffer_registry.find(m.buffer_id);
    assert(search != buffer_registry.end());
    return (void *) (*search).second->peer_va();
}

void KnappComputeDevice::memwrite(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    const auto &search = buffer_registry.find(host_buf.buffer_id);
    assert(search != buffer_registry.end());
    (*search).second->write(offset, size, true);
}

void KnappComputeDevice::memread(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    const auto &search = buffer_registry.find(host_buf.buffer_id);
    assert(search != buffer_registry.end());
    (*search).second->read(offset, size, true);
}

// vim: ts=8 sts=4 sw=4 et
