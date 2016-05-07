#include <nba/core/intrinsic.hh>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/hosttypes.hh>
#include <nba/engines/knapp/hostutils.hh>
#include <nba/engines/knapp/sharedutils.hh>
#include <nba/engines/knapp/computecontext.hh>
#include <nba/engines/knapp/computedevice.hh>
#include <rte_memzone.h>
#include <rte_memory.h>
#include <unistd.h>
#include <scif.h>

using namespace std;
using namespace nba;

struct cuda_event_context {
    ComputeContext *computectx;
    void (*callback)(ComputeContext *ctx, void *user_arg);
    void *user_arg;
};

#define IO_BASE_SIZE (16 * 1024 * 1024)
#define IO_MEMPOOL_ALIGN (8lu)

KnappComputeContext::KnappComputeContext(unsigned ctx_id, ComputeDevice *mother)
 : ComputeContext(ctx_id, mother), checkbits_d(NULL), checkbits_h(NULL),
   mz(reserve_memory(mother)), num_kernel_args(0)
{
    type_name = "knapp.phi";
    size_t io_base_size = ALIGN_CEIL(IO_BASE_SIZE, getpagesize());
    int rc;

    /* Initialize Knapp vDev Parameters. */
    vdev.device_id = ctx_id;
    vdev.ht_per_core = 2;                   // FIXME: set from config
    vdev.pipeline_depth = NBA_MAX_IO_BASES; // Key adaptation: app-specific I/O buffers -> io_base_t
    const unsigned num_cores_per_vdev = 2;  // FIXME: set from config

    /* TODO: Create vDevice. */
    // We don't have know which MIC cores vDevice is using.
    // It is just matter of MIC-side daemon.

    /* Initialize vDev communication channels. */
    vdev.master_port = (dynamic_cast<KnappComputeDevice*>(mother))->master_port;
    vdev.data_epd = scif_open();
    if (vdev.data_epd == SCIF_OPEN_FAILED)
        rte_exit(EXIT_FAILURE, "scif_open() for data_epd failed.");
    vdev.ctrl_epd = scif_open();
    if (vdev.ctrl_epd == SCIF_OPEN_FAILED)
        rte_exit(EXIT_FAILURE, "scif_open() for ctrl_epd failed.");
    vdev.mic_data_port.node = knapp::remote_scif_nodes[0];
    vdev.mic_data_port.port = knapp::get_mic_data_port(ctx_id);
    rc = scif_connect(vdev.data_epd, &vdev.mic_data_port);
    assert(0 < rc);

    /* TODO: Create pollrings & RMA buffers. */

    vdev.next_poll = 0;

    /* Prepare offload task structures. */
    vdev.tasks_in_flight = (struct knapp::offload_task *) rte_zmalloc_socket(nullptr,
            sizeof(struct knapp::offload_task) * vdev.pipeline_depth,
            CACHE_LINE_SIZE, node_id);
    assert(vdev.tasks_in_flight != nullptr);

    /* Initialize I/O buffers. */
    NEW(node_id, io_base_ring, FixedRing<unsigned>, NBA_MAX_IO_BASES, node_id);
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        io_base_ring->push_back(i);
        //NEW(node_id, _cuda_mempool_in[i], KnappMemoryPool, io_base_size, IO_MEMPOOL_ALIGN);
        //NEW(node_id, _cuda_mempool_out[i], KnappMemoryPool, io_base_size, IO_MEMPOOL_ALIGN);
        //_cuda_mempool_in[i]->init();
        //_cuda_mempool_out[i]->init();
        NEW(node_id, _cpu_mempool_in[i], CPUMemoryPool, io_base_size, IO_MEMPOOL_ALIGN, 0);
        NEW(node_id, _cpu_mempool_out[i], CPUMemoryPool, io_base_size, IO_MEMPOOL_ALIGN, 0);
        //_cpu_mempool_in[i]->init_with_flags(nullptr, cudaHostAllocPortable);
        //_cpu_mempool_out[i]->init_with_flags(nullptr, cudaHostAllocPortable);
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
        //_cuda_mempool_in[i]->destroy();
        //_cuda_mempool_out[i]->destroy();
        _cpu_mempool_in[i]->destroy();
        _cpu_mempool_out[i]->destroy();
    }
    if (mz != nullptr)
        rte_memzone_free(mz);
    scif_close(vdev.data_epd);
    scif_close(vdev.ctrl_epd);
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
    assert(0 == _cpu_mempool_in[i]->alloc(size, host_mem));
    //assert(0 == _cuda_mempool_in[i]->alloc(size, dev_mem));
    // for debugging
    //assert(((uintptr_t)host_mem.ptr & 0xffff) == ((uintptr_t)dev_mem.ptr & 0xffff));
    return 0;
}

int KnappComputeContext::alloc_output_buffer(io_base_t io_base, size_t size,
                                            host_mem_t &host_mem, dev_mem_t &dev_mem)
{
    unsigned i = io_base;
    assert(0 == _cpu_mempool_out[i]->alloc(size, host_mem));
    //assert(0 == _cuda_mempool_out[i]->alloc(size, dev_mem));
    // for debugging
    //assert(((uintptr_t)host_mem.ptr & 0xffff) == ((uintptr_t)dev_mem.ptr & 0xffff));
    return 0;
}

void KnappComputeContext::map_input_buffer(io_base_t io_base, size_t offset, size_t len,
                                          host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.ptr = (void *) ((uintptr_t) _cpu_mempool_in[i]->get_base_ptr().ptr + offset);
    //dbuf.ptr = (void *) ((uintptr_t) _cuda_mempool_in[i]->get_base_ptr().ptr + offset);
    // len is ignored.
}

void KnappComputeContext::map_output_buffer(io_base_t io_base, size_t offset, size_t len,
                                           host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.ptr = (void *) ((uintptr_t) _cpu_mempool_out[i]->get_base_ptr().ptr + offset);
    //dbuf.ptr = (void *) ((uintptr_t) _cuda_mempool_out[i]->get_base_ptr().ptr + offset);
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
    return _cpu_mempool_in[i]->get_alloc_size();
}

size_t KnappComputeContext::get_output_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_out[i]->get_alloc_size();
}

void KnappComputeContext::clear_io_buffers(io_base_t io_base)
{
    unsigned i = io_base;
    _cpu_mempool_in[i]->reset();
    _cpu_mempool_out[i]->reset();
    //_cuda_mempool_in[i]->reset();
    //_cuda_mempool_out[i]->reset();
    io_base_ring->push_back(i);
}

int KnappComputeContext::enqueue_memwrite_op(const host_mem_t host_buf,
                                            const dev_mem_t dev_buf,
                                            size_t offset, size_t size)
{
    /* TODO: scif_writeto() to the vDevice's input RMA. */
    return 0;
}

int KnappComputeContext::enqueue_memread_op(const host_mem_t host_buf,
                                           const dev_mem_t dev_buf,
                                           size_t offset, size_t size)
{
    /* TODO: scif_send() the params to the vDevice's master. */
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
    // TODO: implement?
}

int KnappComputeContext::enqueue_kernel_launch(dev_kernel_t kernel, struct resource_param *res)
{
    if (unlikely(res->num_workgroups == 0))
        res->num_workgroups = 1;
    void *raw_args[num_kernel_args];
    for (unsigned i = 0; i < num_kernel_args; i++) {
        raw_args[i] = kernel_args[i].ptr;
    }
    state = ComputeContext::RUNNING;
    //cutilSafeCall(cudaLaunchKernel(kernel.ptr, dim3(res->num_workgroups),
    //                               dim3(res->num_threads_per_workgroup),
    //                               (void **) &raw_args[0], 1024, _stream));
    // TODO: scif_fence_signal() to the vDevice's input pollring.
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
    // TODO: compiler_fence() + check vDevice's output pollring.
    //vdev.poll_ring->wait();
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
