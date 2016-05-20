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

    /* Initialize poll ring. */
    PollRing *pollring;
    NEW(node_id, pollring, PollRing, vdev.data_epd, NBA_MAX_IO_BASES, node_id);
    request.set_type(CtrlRequest::CREATE_POLLRING);
    CtrlRequest::PollRingParam *ring_param = request.mutable_pollring();
    ring_param->set_vdev_handle((uintptr_t) vdev.handle);
    ring_param->set_ring_id(0);
    ring_param->set_len(NBA_MAX_IO_BASES);
    ring_param->set_local_ra((uint64_t) pollring->ra());
    ctrl_invoke(ctrl_epd, request, response);
    assert(CtrlResponse::SUCCESS == response.reply());
    pollring->set_peer_ra(response.resource().peer_ra());
    vdev.poll_rings[0] = pollring;

    /* Initialize I/O buffers. */
    CtrlRequest::RMABufferParam *rma_param;
    NEW(node_id, io_base_ring, FixedRing<unsigned>, NBA_MAX_IO_BASES, node_id);
    next_task_id = 0;
    //NEW(node_id, task_id_ring, FixedRing<unsigned>, NBA_MAX_IO_BASES, node_id);
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        io_base_ring->push_back(i);
        //task_id_ring->push_back(i);
        RMABuffer *rma_inbuf, *rma_outbuf;
        NEW(node_id, rma_inbuf, RMABuffer, vdev.data_epd, io_base_size, node_id);
        NEW(node_id, rma_outbuf, RMABuffer, vdev.data_epd, io_base_size, node_id);

        request.Clear();
        request.set_type(CtrlRequest::CREATE_RMABUFFER);
        rma_param = request.mutable_rma();
        rma_param->set_vdev_handle((uintptr_t) vdev.handle);
        rma_param->set_buffer_id(compose_buffer_id(false, i, INPUT));
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
        rma_param->set_buffer_id(compose_buffer_id(false, i, OUTPUT));
        rma_param->set_size(io_base_size);
        rma_param->set_local_ra((uint64_t) rma_outbuf->ra());
        ctrl_invoke(ctrl_epd, request, response);
        assert(CtrlResponse::SUCCESS == response.reply());
        rma_outbuf->set_peer_ra(response.resource().peer_ra());
        rma_outbuf->set_peer_va(response.resource().peer_va());

        NEW(node_id, _local_mempool_in[i], RMALocalMemoryPool, i, rma_inbuf, io_base_size);
        NEW(node_id, _local_mempool_out[i], RMALocalMemoryPool, i, rma_outbuf, io_base_size);
        NEW(node_id, _local_mempool_inout[i], RMALocalMemoryPool, i, rma_inbuf, io_base_size);
        NEW(node_id, _peer_mempool_in[i], RMAPeerMemoryPool, i, rma_inbuf, io_base_size);
        NEW(node_id, _peer_mempool_out[i], RMAPeerMemoryPool, i, rma_outbuf, io_base_size);
        NEW(node_id, _peer_mempool_inout[i], RMAPeerMemoryPool, i, rma_inbuf, io_base_size);
        _local_mempool_in[i]->init();
        _local_mempool_out[i]->init();
        _local_mempool_inout[i]->init();
        _peer_mempool_in[i]->init();
        _peer_mempool_out[i]->init();
        _peer_mempool_inout[i]->init();
    }

    size_t aligned_tp_size = ALIGN_CEIL(sizeof(struct taskitem) * NBA_MAX_IO_BASES, PAGE_SIZE);
    size_t aligned_dp_size = ALIGN_CEIL(sizeof(struct d2hcopy) * NBA_MAX_IO_BASES, PAGE_SIZE);
    NEW(node_id, vdev.task_params, RMABuffer,
        vdev.data_epd, aligned_tp_size, node_id);
    NEW(node_id, vdev.d2h_params, RMABuffer,
        vdev.data_epd, aligned_dp_size, node_id);

    request.Clear();
    request.set_type(CtrlRequest::CREATE_RMABUFFER);
    rma_param = request.mutable_rma();
    rma_param->set_vdev_handle((uintptr_t) vdev.handle);
    rma_param->set_buffer_id(BUFFER_TASK_PARAMS);
    rma_param->set_size(aligned_tp_size);
    rma_param->set_local_ra((uint64_t) vdev.task_params->ra());
    ctrl_invoke(ctrl_epd, request, response);
    assert(CtrlResponse::SUCCESS == response.reply());
    vdev.task_params->set_peer_ra(response.resource().peer_ra());
    vdev.task_params->set_peer_va(response.resource().peer_va());

    request.Clear();
    request.set_type(CtrlRequest::CREATE_RMABUFFER);
    rma_param = request.mutable_rma();
    rma_param->set_vdev_handle((uintptr_t) vdev.handle);
    rma_param->set_buffer_id(BUFFER_D2H_PARAMS);
    rma_param->set_size(aligned_dp_size);
    rma_param->set_local_ra((uint64_t) vdev.d2h_params->ra());
    ctrl_invoke(ctrl_epd, request, response);
    assert(CtrlResponse::SUCCESS == response.reply());
    vdev.d2h_params->set_peer_ra(response.resource().peer_ra());
    vdev.d2h_params->set_peer_va(response.resource().peer_va());
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
        _local_mempool_inout[i]->init();
        _peer_mempool_in[i]->destroy();
        _peer_mempool_out[i]->destroy();
        _local_mempool_in[i]->rma_buffer()->~RMABuffer();
        _local_mempool_out[i]->rma_buffer()->~RMABuffer();
        rte_free(_local_mempool_in[i]->rma_buffer());
        rte_free(_local_mempool_out[i]->rma_buffer());
        rte_free(_local_mempool_in[i]);
        rte_free(_local_mempool_out[i]);
        rte_free(_local_mempool_inout[i]);
        rte_free(_peer_mempool_in[i]->rma_buffer());
        rte_free(_peer_mempool_out[i]->rma_buffer());
        rte_free(_peer_mempool_in[i]);
        rte_free(_peer_mempool_out[i]);
        rte_free(_peer_mempool_inout[i]);
    }
    vdev.task_params->~RMABuffer();
    vdev.d2h_params->~RMABuffer();
    rte_free(vdev.task_params);
    rte_free(vdev.d2h_params);
    if (mz != nullptr)
        rte_memzone_free(mz);

    vdev.poll_rings[0]->~PollRing();
    rte_free(vdev.poll_rings[0]);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::DESTROY_VDEV);
    request.mutable_resource()->set_handle((uintptr_t) vdev.handle);
    ctrl_invoke(ctrl_epd, request, response);
    assert(CtrlResponse::SUCCESS == response.reply());

    scif_close(vdev.data_epd);
    rte_free(vdev.tasks_in_flight);
}

uint32_t KnappComputeContext::alloc_task_id()
{
    unsigned t = next_task_id;
    next_task_id = (next_task_id + 1) % NBA_MAX_IO_BASES;
    return t;
}

void KnappComputeContext::release_task_id(uint32_t task_id)
{
    // do nothing
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

int KnappComputeContext::alloc_inout_buffer(io_base_t io_base, size_t size,
                                            host_mem_t &host_mem, dev_mem_t &dev_mem)
{
    unsigned i = io_base;
    host_mem_t hi, hio;
    dev_mem_t di, dio;
    assert(0 == _local_mempool_in[i]->alloc(size, hi));
    assert(0 == _peer_mempool_in[i]->alloc(size, di));
    assert(0 == _local_mempool_inout[i]->alloc(size, hio));
    assert(0 == _peer_mempool_inout[i]->alloc(size, dio));
    assert(hi.m.unwrap_ptr == hio.m.unwrap_ptr);
    assert(di.m.unwrap_ptr == dio.m.unwrap_ptr);
    host_mem = hi;
    dev_mem  = di;
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

void KnappComputeContext::get_input_buffer(io_base_t io_base,
                                           host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.m = {
        compose_buffer_id(false, io_base, INPUT),
        (void *) ((uintptr_t) _local_mempool_in[i]->get_base_ptr().m.unwrap_ptr)
    };
    dbuf.m = {
        compose_buffer_id(false, io_base, INPUT),
        (void *) ((uintptr_t) _peer_mempool_in[i]->get_base_ptr().m.unwrap_ptr)
    };
}

void KnappComputeContext::get_inout_buffer(io_base_t io_base,
                                           host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.m = {
        compose_buffer_id(false, io_base, OUTPUT),
        (void *) ((uintptr_t) _local_mempool_inout[i]->get_base_ptr().m.unwrap_ptr)
    };
    dbuf.m = {
        compose_buffer_id(false, io_base, OUTPUT),
        (void *) ((uintptr_t) _peer_mempool_inout[i]->get_base_ptr().m.unwrap_ptr)
    };
}

void KnappComputeContext::get_output_buffer(io_base_t io_base,
                                            host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.m = {
        compose_buffer_id(false, io_base, OUTPUT),
        (void *) ((uintptr_t) _local_mempool_out[i]->get_base_ptr().m.unwrap_ptr)
    };
    dbuf.m = {
        compose_buffer_id(false, io_base, OUTPUT),
        (void *) ((uintptr_t) _peer_mempool_out[i]->get_base_ptr().m.unwrap_ptr)
    };
}

void *KnappComputeContext::unwrap_host_buffer(const host_mem_t hbuf) const
{
    return hbuf.m.unwrap_ptr;
}

void *KnappComputeContext::unwrap_device_buffer(const dev_mem_t dbuf) const
{
    return dbuf.m.unwrap_ptr;
}

size_t KnappComputeContext::get_input_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _local_mempool_in[i]->get_alloc_size();
}

size_t KnappComputeContext::get_inout_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _local_mempool_inout[i]->get_alloc_size();
}

size_t KnappComputeContext::get_output_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _local_mempool_out[i]->get_alloc_size();
}

void KnappComputeContext::shift_inout_base(io_base_t io_base, size_t len)
{
    unsigned i = io_base;
    _local_mempool_inout[i]->shift_base(len);
    _peer_mempool_inout[i]->shift_base(len);
}

void KnappComputeContext::clear_io_buffers(io_base_t io_base)
{
    unsigned i = io_base;
    _local_mempool_in[i]->reset();
    _local_mempool_out[i]->reset();
    _local_mempool_inout[i]->reset();
    _peer_mempool_in[i]->reset();
    _peer_mempool_out[i]->reset();
    _peer_mempool_inout[i]->reset();
    io_base_ring->push_back(i);
    //fprintf(stderr, "cctx[%u] clear_iobuf: io_base %u\n", ctx_id, io_base);
}

int KnappComputeContext::enqueue_memwrite_op(uint32_t task_id,
                                             const host_mem_t hbuf,
                                             const dev_mem_t dbuf,
                                             size_t offset, size_t size)
{
    assert(hbuf.m.buffer_id == dbuf.m.buffer_id);
    bool is_global;
    uint32_t io_base;
    rma_direction dir;
    std::tie(is_global, io_base, dir) = decompose_buffer_id(hbuf.m.buffer_id);
    assert(!is_global);
    //fprintf(stderr, "cctx[%u] enqueue_memwrite: task_id %u, io_base %u\n", ctx_id, task_id, io_base);
    RMABuffer *b = _local_mempool_in[io_base]->rma_buffer();
    b->write(offset, size, false);
    return 0;
}

int KnappComputeContext::enqueue_memread_op(uint32_t task_id,
                                            const host_mem_t hbuf,
                                            const dev_mem_t dbuf,
                                            size_t offset, size_t size)
{
    assert(hbuf.m.buffer_id == dbuf.m.buffer_id);
    bool is_global;
    uint32_t io_base;
    rma_direction dir;
    std::tie(is_global, io_base, dir) = decompose_buffer_id(hbuf.m.buffer_id);
    assert(!is_global);
    //fprintf(stderr, "cctx[%u] enqueue_memread: task_id %u, io_base %u, offset %u, size %u\n",
    //        ctx_id, task_id, io_base, offset, size);
    RMABuffer *cb = vdev.d2h_params;
    struct d2hcopy &c = reinterpret_cast<struct d2hcopy *>(cb->va())[task_id];
    uint16_t i = (c.num_copies ++);
    c.buffer_id[i] = hbuf.m.buffer_id;
    c.offset[i] = (uint32_t) offset;
    c.size[i] = (uint32_t) size;
    return 0;
}

void KnappComputeContext::h2d_done(uint32_t task_id)
{
    RMABuffer *cb = vdev.d2h_params;
    struct d2hcopy &c = reinterpret_cast<struct d2hcopy *>(cb->va())[task_id];
    c.num_copies = 0;
}

void KnappComputeContext::d2h_done(uint32_t task_id)
{
    RMABuffer *cb = vdev.d2h_params;
    cb->write(sizeof(struct d2hcopy) * task_id, sizeof(struct d2hcopy), false);
    vdev.poll_rings[0]->remote_notify(task_id, KNAPP_H2D_COMPLETE);
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

int KnappComputeContext::enqueue_kernel_launch(
// FIXME: add task_id parameter like other enqueue_*() methods
        dev_kernel_t kernel,
        struct resource_param *res)
{
    if (unlikely(res->num_workgroups == 0))
        res->num_workgroups = 1;
    // FIXME: merge with h2d datablock copy
    RMABuffer *tb = vdev.task_params;
    uint32_t task_id = res->task_id;
    struct taskitem &t = reinterpret_cast<struct taskitem *>(tb->va())[task_id];
    t.task_id = task_id;
    t.kernel_id = kernel.kernel_id;
    t.num_items = res->num_workitems;
    t.num_args = num_kernel_args;
    for (unsigned i = 0; i < num_kernel_args; i++) {
        memcpy(&t.args[i], kernel_args[i].ptr, kernel_args[i].size);
    }
    state = ComputeContext::RUNNING;
    //fprintf(stderr, "cctx[%u] enqueue_kernel_launch: task_id: %u, num_items: %u, args: %u\n",
    //        ctx_id, task_id, t.num_items, t.num_args);
    tb->write(sizeof(struct taskitem) * task_id, sizeof(struct taskitem), false);
    return 0;
}

bool KnappComputeContext::poll_input_finished(uint32_t task_id)
{
    /* Proceed to kernel launch without waiting. */
    return true;
}

bool KnappComputeContext::poll_kernel_finished(uint32_t task_id)
{
    /* Proceed to D2H copy initiation without waiting. */
    return true;
}

bool KnappComputeContext::poll_output_finished(uint32_t task_id)
{
    return vdev.poll_rings[0]->poll(task_id, KNAPP_D2H_COMPLETE);
}

int KnappComputeContext::enqueue_event_callback(
        uint32_t task_id,
        void (*func_ptr)(ComputeContext *ctx, void *user_arg),
        void *user_arg)
{
    /* Not implemented. */
    return 0;
}


// vim: ts=8 sts=4 sw=4 et
