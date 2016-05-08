#include <nba/core/intrinsic.hh>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/hosttypes.hh>
#include <nba/engines/knapp/hostutils.hh>
#include <nba/engines/knapp/sharedutils.hh>
#include <nba/engines/knapp/computecontext.hh>
#include <nba/engines/knapp/computedevice.hh>
#include <nba/engines/knapp/rma.hh>
#include <nba/engines/knapp/mempool.hh>
#include <nba/engines/knapp/pollring.hh>
#include <rte_memzone.h>
#include <rte_memory.h>
#include <unistd.h>
#include <scif.h>

using namespace nba;
using namespace nba::knapp;

struct cuda_event_context {
    ComputeContext *computectx;
    void (*callback)(ComputeContext *ctx, void *user_arg);
    void *user_arg;
};

#define IO_BASE_SIZE (16 * 1024 * 1024)
#define IO_MEMPOOL_ALIGN (8lu)

KnappComputeContext::KnappComputeContext(unsigned ctx_id, ComputeDevice *mother)
 : ComputeContext(ctx_id, mother),
   mz(reserve_memory(mother)), num_kernel_args(0)
{
    type_name = "knapp.phi";
    size_t io_base_size = ALIGN_CEIL(IO_BASE_SIZE, getpagesize());
    int rc;

    /* Initialize Knapp vDev Parameters. */
    vdev.ht_per_core = 2;                   // FIXME: set from config
    vdev.pipeline_depth = NBA_MAX_IO_BASES; // Key adaptation: app-specific I/O buffers -> io_base_t
    const unsigned num_cores_per_vdev = 4;  // FIXME: set from config

    /* Take ctrl_epd from the mother device. */
    ctrl_epd = (dynamic_cast<KnappComputeDevice*>(mother))->ctrl_epd;
    vdev.ctrl_epd = ctrl_epd;

    /* Create a vDevice. */
    // We don't have to know which MIC cores vDevice is using.
    // It is just matter of MIC-side daemon.
    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::CREATE_VDEV);
    CtrlRequest::vDeviceInfoParam *v = request.mutable_vdevinfo();
    // NOTE: Junhyun used 2x2 with vectorization while Keunhong used 8x2.
    v->set_num_pcores(2);
    v->set_num_lcores_per_pcore(2);
    v->set_pipeline_depth(32);
    ctrl_invoke(ctrl_epd, request, response);
    assert(CtrlResponse::SUCCESS == response.reply());
    vdev.handle    = (void *) response.resource().handle();
    vdev.device_id = response.resource().id();

    /* Initialize the vDev data channel. */
    vdev.data_epd = scif_open();
    if (vdev.data_epd == SCIF_OPEN_FAILED)
        rte_exit(EXIT_FAILURE, "scif_open() for data_epd failed.");
    vdev.mic_data_port.node = remote_scif_nodes[0];
    vdev.mic_data_port.port = get_mic_data_port(vdev.device_id);
    rc = scif_connect(vdev.data_epd, &vdev.mic_data_port);
    assert(0 < rc);
    vdev.next_poll = 0;

    /* Prepare offload task structures. */
    vdev.tasks_in_flight = (struct offload_task *) rte_zmalloc_socket(nullptr,
            sizeof(struct offload_task) * vdev.pipeline_depth,
            CACHE_LINE_SIZE, node_id);
    assert(vdev.tasks_in_flight != nullptr);

    /* Initialize I/O buffers. */
    NEW(node_id, io_base_ring, FixedRing<unsigned>, NBA_MAX_IO_BASES, node_id);
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        io_base_ring->push_back(i);
        RMABuffer *rma_inbuf, *rma_outbuf;
        CtrlRequest::RMABufferParam *rma_param;
        NEW(node_id, rma_inbuf, RMABuffer, vdev.data_epd, io_base_size, node_id);
        NEW(node_id, rma_outbuf, RMABuffer, vdev.data_epd, io_base_size, node_id);

        request.Clear();
        request.set_type(CtrlRequest::CREATE_RMABUFFER);
        rma_param = request.mutable_rma();
        rma_param->set_vdev_handle((uintptr_t) vdev.handle);
        // TODO: combine io_base ID with ID_INPUT.
        rma_param->set_buffer_id(ID_INPUT);
        rma_param->set_size(io_base_size);
        rma_param->set_local_ra((uint64_t) rma_inbuf->ra());
        ctrl_invoke(ctrl_epd, request, response);
        assert(CtrlResponse::SUCCESS == response.reply());
        rma_inbuf->set_peer_ra(response.resource().peer_ra());
        rma_inbuf->set_peer_va(response.resource().peer_va());

        request.Clear();
        request.set_type(CtrlRequest::CREATE_RMABUFFER);
        rma_param = request.mutable_rma();
        rma_param->set_vdev_handle((uintptr_t) vdev.handle);
        // TODO: combine io_base ID with ID_OUTPUT.
        rma_param->set_buffer_id(ID_OUTPUT);
        rma_param->set_size(io_base_size);
        rma_param->set_local_ra((uint64_t) rma_outbuf->ra());
        ctrl_invoke(ctrl_epd, request, response);
        assert(CtrlResponse::SUCCESS == response.reply());
        rma_outbuf->set_peer_ra(response.resource().peer_ra());
        rma_outbuf->set_peer_va(response.resource().peer_va());

        NEW(node_id, _local_mempool_in[i], RMALocalMemoryPool, rma_inbuf);
        NEW(node_id, _local_mempool_out[i], RMALocalMemoryPool, rma_outbuf);
        NEW(node_id, _peer_mempool_in[i], RMAPeerMemoryPool, rma_inbuf);
        NEW(node_id, _peer_mempool_out[i], RMAPeerMemoryPool, rma_outbuf);
        _local_mempool_in[i]->init();
        _local_mempool_out[i]->init();
        _peer_mempool_in[i]->init();
        _peer_mempool_out[i]->init();
    }
}

const struct rte_memzone *KnappComputeContext::reserve_memory(ComputeDevice *mother)
{
    return nullptr;
}

KnappComputeContext::~KnappComputeContext()
{
    //cutilSafeCall(cudaStreamDestroy(_stream));
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        _local_mempool_in[i]->destroy();
        _local_mempool_out[i]->destroy();
        _peer_mempool_in[i]->destroy();
        _peer_mempool_out[i]->destroy();
        _local_mempool_in[i]->rma_buffer()->~RMABuffer();
        _local_mempool_out[i]->rma_buffer()->~RMABuffer();
        rte_free(_local_mempool_in[i]->rma_buffer());
        rte_free(_local_mempool_out[i]->rma_buffer());
        rte_free(_local_mempool_in[i]);
        rte_free(_local_mempool_out[i]);
        rte_free(_peer_mempool_in[i]->rma_buffer());
        rte_free(_peer_mempool_out[i]->rma_buffer());
        rte_free(_peer_mempool_in[i]);
        rte_free(_peer_mempool_out[i]);
    }
    if (mz != nullptr)
        rte_memzone_free(mz);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::DESTROY_VDEV);
    request.mutable_resource()->set_handle((uintptr_t) vdev.handle);
    ctrl_invoke(ctrl_epd, request, response);
    assert(CtrlResponse::SUCCESS == response.reply());

    scif_close(vdev.data_epd);
    rte_free(vdev.tasks_in_flight);
}

io_base_t KnappComputeContext::alloc_io_base()
{
    if (io_base_ring->empty()) return INVALID_IO_BASE;
    unsigned i = io_base_ring->front();
    io_base_ring->pop_front();
    return (io_base_t) i;
}

int KnappComputeContext::alloc_input_buffer(io_base_t io_base, size_t size,
                                           host_mem_t &host_mem, dev_mem_t &dev_mem)
{
    unsigned i = io_base;
    assert(0 == _local_mempool_in[i]->alloc(size, host_mem));
    assert(0 == _peer_mempool_in[i]->alloc(size, dev_mem));
    return 0;
}

int KnappComputeContext::alloc_output_buffer(io_base_t io_base, size_t size,
                                            host_mem_t &host_mem, dev_mem_t &dev_mem)
{
    unsigned i = io_base;
    assert(0 == _local_mempool_out[i]->alloc(size, host_mem));
    assert(0 == _peer_mempool_out[i]->alloc(size, dev_mem));
    return 0;
}

void KnappComputeContext::map_input_buffer(io_base_t io_base, size_t offset, size_t len,
                                          host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.ptr = (void *) ((uintptr_t) _local_mempool_in[i]->get_base_ptr().ptr + offset);
    dbuf.ptr = (void *) ((uintptr_t) _peer_mempool_in[i]->get_base_ptr().ptr + offset);
    // len is ignored.
}

void KnappComputeContext::map_output_buffer(io_base_t io_base, size_t offset, size_t len,
                                           host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.ptr = (void *) ((uintptr_t) _local_mempool_out[i]->get_base_ptr().ptr + offset);
    dbuf.ptr = (void *) ((uintptr_t) _peer_mempool_out[i]->get_base_ptr().ptr + offset);
    // len is ignored.
}

void *KnappComputeContext::unwrap_host_buffer(const host_mem_t hbuf) const
{
    return hbuf.ptr;
}

void *KnappComputeContext::unwrap_device_buffer(const dev_mem_t dbuf) const
{
    return dbuf.ptr;
}

size_t KnappComputeContext::get_input_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _local_mempool_in[i]->get_alloc_size();
}

size_t KnappComputeContext::get_output_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _local_mempool_out[i]->get_alloc_size();
}

void KnappComputeContext::clear_io_buffers(io_base_t io_base)
{
    unsigned i = io_base;
    _local_mempool_in[i]->reset();
    _local_mempool_out[i]->reset();
    _peer_mempool_in[i]->reset();
    _peer_mempool_out[i]->reset();
    io_base_ring->push_back(i);
}

int KnappComputeContext::enqueue_memwrite_op(const host_mem_t host_buf,
                                            const dev_mem_t dev_buf,
                                            size_t offset, size_t size)
{
    // TODO: retrieve relevant RMALocalMemoryPool from host_buf.
    // cur_task_id == io_base ID.
    // TODO: get rma_buffer() and call ->write(offset, size);
    // this may be deferred to enqueue_kernel_laucn()
    // to send along with taskitem.
    return 0;
}

int KnappComputeContext::enqueue_memread_op(const host_mem_t host_buf,
                                           const dev_mem_t dev_buf,
                                           size_t offset, size_t size)
{
    // TODO: retrieve relevant RMALocalMemoryPool from host_buf.
    // TODO: scif_send() the params to via vdev.data_epd. */
    return 0;
}

void KnappComputeContext::clear_kernel_args()
{
    num_kernel_args = 0;
}

void KnappComputeContext::push_kernel_arg(struct kernel_arg &arg)
{
    assert(num_kernel_args < KNAPP_MAX_KERNEL_ARGS);
    kernel_args[num_kernel_args ++] = arg;  /* Copied to the array. */
}

void KnappComputeContext::push_common_kernel_args()
{
    // do nothing.
}

int KnappComputeContext::enqueue_kernel_launch(dev_kernel_t kernel, struct resource_param *res)
{
    if (unlikely(res->num_workgroups == 0))
        res->num_workgroups = 1;
    // TODO: initialize taskitem information. (kernel ID, arguments)
    //void *raw_args[num_kernel_args];
    //for (unsigned i = 0; i < num_kernel_args; i++) {
    //    raw_args[i] = kernel_args[i].ptr;
    //}
    state = ComputeContext::RUNNING;

    // TODO: scif_fence_signal() to the vDevice's input pollring.
    vdev.poll_rings[0]->remote_notify(cur_task_id, KNAPP_H2D_COMPLETE);
    return 0;
}

bool KnappComputeContext::poll_input_finished()
{
    /* Proceed to kernel launch without waiting. */
    return true;
}

bool KnappComputeContext::poll_kernel_finished()
{
    /* Proceed to D2H copy initiation without waiting. */
    return true;
}

bool KnappComputeContext::poll_output_finished()
{
    vdev.poll_rings[0]->wait(cur_task_id, KNAPP_D2H_COMPLETE);
    return true;
}

int KnappComputeContext::enqueue_event_callback(
        void (*func_ptr)(ComputeContext *ctx, void *user_arg),
        void *user_arg)
{
    /* Not implemented. */
    return 0;
}


// vim: ts=8 sts=4 sw=4 et
