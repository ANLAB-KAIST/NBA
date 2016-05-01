#include <nba/core/intrinsic.hh>
#include <nba/framework/logging.hh>
#include <nba/engines/knapp/types.hh>
#include <nba/engines/knapp/utils.hh>
#include <nba/engines/knapp/computedevice.hh>
#include <rte_memory.h>
#include <scif.h>

using namespace std;
using namespace nba;

KnappComputeDevice::KnappComputeDevice(
        unsigned node_id, unsigned device_id, size_t num_contexts
) : ComputeDevice(node_id, device_id, num_contexts)
{
    type_name = "knapp.phi";
    assert(num_contexts > 0);
    RTE_LOG(DEBUG, COPROC, "KnappComputeDevice: # contexts: %lu\n", num_contexts);
    for (unsigned i = 0; i < num_contexts; i++) {
        KnappComputeContext *ctx = nullptr;
        NEW(node_id, ctx, KnappComputeContext, i, this);
        _ready_contexts.push_back(ctx);
        contexts.push_back((ComputeContext *) ctx);
    }
}

KnappComputeDevice::~KnappComputeDevice()
{
    for (auto it = _ready_contexts.begin(); it != _ready_contexts.end(); it++) {
        KnappComputeContext *ctx = *it;
        delete ctx;
        *it = NULL;
    }
    for (auto it = _active_contexts.begin(); it != _active_contexts.end(); it++) {
        KnappComputeContext *ctx = *it;
        delete ctx;
        *it = NULL;
    }
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
    void *ptr = nullptr;
    //int nvflags = 0;
    //nvflags |= (flags & HOST_PINNED) ? cudaHostAllocPortable : 0;
    //nvflags |= (flags & HOST_MAPPED) ? cudaHostAllocMapped : 0;
    //nvflags |= (flags & HOST_WRITECOMBINED) ? cudaHostAllocWriteCombined : 0;
    //cutilSafeCall(cudaHostAlloc(&ptr, size, nvflags));
    assert(ptr != nullptr);
    return { ptr };
}

dev_mem_t KnappComputeDevice::alloc_device_buffer(size_t size, int flags)
{
    void *ptr = nullptr;
    //cutilSafeCall(cudaMalloc(&ptr, size));
    assert(ptr != nullptr);
    return { ptr };
}

void KnappComputeDevice::free_host_buffer(host_mem_t m)
{
    //cutilSafeCall(cudaFreeHost(m.ptr));
}

void KnappComputeDevice::free_device_buffer(dev_mem_t m)
{
    //cutilSafeCall(cudaFree(m.ptr));
}

void KnappComputeDevice::memwrite(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    //cutilSafeCall(cudaMemcpy((uint8_t *) dev_buf.ptr + offset, host_buf.ptr, size, cudaMemcpyHostToDevice));
}

void KnappComputeDevice::memread(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    //cutilSafeCall(cudaMemcpy(host_buf.ptr, (uint8_t *) dev_buf.ptr + offset, size, cudaMemcpyDeviceToHost));
}

// vim: ts=8 sts=4 sw=4 et
