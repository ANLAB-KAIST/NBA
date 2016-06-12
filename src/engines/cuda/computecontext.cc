#include <nba/core/intrinsic.hh>
#include <nba/engines/cuda/computecontext.hh>
#include <nba/engines/cuda/mempool.hh>
#include <rte_memzone.h>
#include <unistd.h>

using namespace std;
using namespace nba;

struct cuda_event_context {
    ComputeContext *computectx;
    void (*callback)(ComputeContext *ctx, void *user_arg);
    void *user_arg;
};

#define IO_BASE_SIZE (16 * 1024 * 1024)
#define IO_MEMPOOL_ALIGN (8lu)

CUDAComputeContext::CUDAComputeContext(unsigned ctx_id, ComputeDevice *mother)
 : ComputeContext(ctx_id, mother), checkbits_d(NULL), checkbits_h(NULL),
   mz(reserve_memory(mother)), num_kernel_args(0)
   /* NOTE: Write-combined memory degrades performance to half... */
{
    type_name = "cuda";
    size_t io_base_size = ALIGN_CEIL(IO_BASE_SIZE, getpagesize());
    cutilSafeCall(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
    NEW(node_id, io_base_ring, FixedRing<unsigned>,
        NBA_MAX_IO_BASES, node_id);
    next_task_id = 0;
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        io_base_ring->push_back(i);
        NEW(node_id, _cuda_mempool_in[i], CUDAMemoryPool, io_base_size, IO_MEMPOOL_ALIGN);
        NEW(node_id, _cuda_mempool_inout[i], CPUMemoryPool, io_base_size, IO_MEMPOOL_ALIGN, 0);
        NEW(node_id, _cuda_mempool_out[i], CUDAMemoryPool, io_base_size, IO_MEMPOOL_ALIGN);
        _cuda_mempool_in[i]->init();
        _cuda_mempool_inout[i]->init_with_flags(_cuda_mempool_in[i]->get_base_ptr().ptr, 0);
        _cuda_mempool_out[i]->init();
        NEW(node_id, _cpu_mempool_in[i], CPUMemoryPool, io_base_size, IO_MEMPOOL_ALIGN, 0);
        NEW(node_id, _cpu_mempool_inout[i], CPUMemoryPool, io_base_size, IO_MEMPOOL_ALIGN, 0);
        NEW(node_id, _cpu_mempool_out[i], CPUMemoryPool, io_base_size, IO_MEMPOOL_ALIGN, 0);
        _cpu_mempool_in[i]->init_with_flags(nullptr, cudaHostAllocPortable);
        _cpu_mempool_out[i]->init_with_flags(nullptr, cudaHostAllocPortable);
        _cpu_mempool_inout[i]->init_with_flags(_cpu_mempool_in[i]->get_base_ptr().ptr, 0);
    }
    cutilSafeCall(cudaHostAlloc((void **) &checkbits_h, MAX_BLOCKS, cudaHostAllocMapped));
    cutilSafeCall(cudaHostGetDevicePointer((void **) &checkbits_d, checkbits_h, 0));
    assert(checkbits_h != NULL);
    assert(checkbits_d != NULL);
    memset(checkbits_h, 0, MAX_BLOCKS);
}

const struct rte_memzone *CUDAComputeContext::reserve_memory(ComputeDevice *mother)
{
    return nullptr;
}

CUDAComputeContext::~CUDAComputeContext()
{
    cutilSafeCall(cudaStreamDestroy(_stream));
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        _cuda_mempool_in[i]->destroy();
        _cuda_mempool_inout[i]->destroy();
        _cuda_mempool_out[i]->destroy();
        _cpu_mempool_in[i]->destroy();
        _cpu_mempool_inout[i]->destroy();
        _cpu_mempool_out[i]->destroy();
    }
    if (mz != nullptr)
        rte_memzone_free(mz);
    cutilSafeCall(cudaFreeHost(checkbits_h));
}

uint32_t CUDAComputeContext::alloc_task_id()
{
    unsigned t = next_task_id;
    next_task_id = (next_task_id + 1) % NBA_MAX_IO_BASES;
    return t;
}

void CUDAComputeContext::release_task_id(uint32_t task_id)
{
    // do nothing
}


io_base_t CUDAComputeContext::alloc_io_base()
{
    if (io_base_ring->empty()) return INVALID_IO_BASE;
    unsigned i = io_base_ring->front();
    io_base_ring->pop_front();
    return (io_base_t) i;
}

int CUDAComputeContext::alloc_input_buffer(io_base_t io_base, size_t size,
                                           host_mem_t &host_mem, dev_mem_t &dev_mem)
{
    unsigned i = io_base;
    assert(0 == _cpu_mempool_in[i]->alloc(size, host_mem));
    assert(0 == _cuda_mempool_in[i]->alloc(size, dev_mem));
    // for debugging
    //assert(((uintptr_t)host_mem.ptr & 0xffff) == ((uintptr_t)dev_mem.ptr & 0xffff));
    return 0;
}

int CUDAComputeContext::alloc_inout_buffer(io_base_t io_base, size_t size,
                                           host_mem_t &host_mem, dev_mem_t &dev_mem)
{
    unsigned i = io_base;
    host_mem_t hi, hio, dio;
    dev_mem_t di;
    assert(0 == _cpu_mempool_in[i]->alloc(size, hi));
    assert(0 == _cpu_mempool_inout[i]->alloc(size, hio));
    assert(0 == _cuda_mempool_in[i]->alloc(size, di));
    assert(0 == _cuda_mempool_inout[i]->alloc(size, dio));
    assert(hi.ptr == hio.ptr);
    assert(di.ptr == dio.ptr);
    host_mem = hi;
    dev_mem  = di;
    // for debugging
    //assert(((uintptr_t)host_mem.ptr & 0xffff) == ((uintptr_t)dev_mem.ptr & 0xffff));
    return 0;
}

int CUDAComputeContext::alloc_output_buffer(io_base_t io_base, size_t size,
                                            host_mem_t &host_mem, dev_mem_t &dev_mem)
{
    unsigned i = io_base;
    assert(0 == _cpu_mempool_out[i]->alloc(size, host_mem));
    assert(0 == _cuda_mempool_out[i]->alloc(size, dev_mem));
    // for debugging
    //assert(((uintptr_t)host_mem.ptr & 0xffff) == ((uintptr_t)dev_mem.ptr & 0xffff));
    return 0;
}

void CUDAComputeContext::get_input_buffer(io_base_t io_base,
                                          host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.ptr = (void *) ((uintptr_t) _cpu_mempool_in[i]->get_base_ptr().ptr);
    dbuf.ptr = (void *) ((uintptr_t) _cuda_mempool_in[i]->get_base_ptr().ptr);
}

void CUDAComputeContext::get_inout_buffer(io_base_t io_base,
                                          host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.ptr = (void *) ((uintptr_t) _cpu_mempool_inout[i]->get_base_ptr().ptr);
    dbuf.ptr = (void *) ((uintptr_t) _cuda_mempool_inout[i]->get_base_ptr().ptr);
}

void CUDAComputeContext::get_output_buffer(io_base_t io_base,
                                           host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.ptr = (void *) ((uintptr_t) _cpu_mempool_out[i]->get_base_ptr().ptr);
    dbuf.ptr = (void *) ((uintptr_t) _cuda_mempool_out[i]->get_base_ptr().ptr);
}

void *CUDAComputeContext::unwrap_host_buffer(const host_mem_t hbuf) const
{
    return hbuf.ptr;
}

void *CUDAComputeContext::unwrap_device_buffer(const dev_mem_t dbuf) const
{
    return dbuf.ptr;
}

size_t CUDAComputeContext::get_input_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_in[i]->get_alloc_size();
}

size_t CUDAComputeContext::get_inout_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_inout[i]->get_alloc_size();
}

size_t CUDAComputeContext::get_output_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_out[i]->get_alloc_size();
}

void CUDAComputeContext::shift_inout_base(io_base_t io_base, size_t len)
{
    unsigned i = io_base;
    _cpu_mempool_inout[i]->shift_base(len);
    _cuda_mempool_inout[i]->shift_base(len);
}

void CUDAComputeContext::clear_io_buffers(io_base_t io_base)
{
    unsigned i = io_base;
    _cpu_mempool_in[i]->reset();
    _cpu_mempool_out[i]->reset();
    _cpu_mempool_inout[i]->reset();
    _cuda_mempool_in[i]->reset();
    _cuda_mempool_out[i]->reset();
    _cuda_mempool_inout[i]->reset();
    io_base_ring->push_back(i);
}

int CUDAComputeContext::enqueue_memwrite_op(uint32_t task_id,
                                            const host_mem_t host_buf,
                                            const dev_mem_t dev_buf,
                                            size_t offset, size_t size)
{
    void *hptr = (void *) ((uintptr_t) host_buf.ptr + offset);
    void *dptr = (void *) ((uintptr_t) dev_buf.ptr + offset);
    cutilSafeCall(cudaMemcpyAsync(dptr, hptr, size,
                                  cudaMemcpyHostToDevice, _stream));
    return 0;
}

int CUDAComputeContext::enqueue_memread_op(uint32_t task_id,
                                           const host_mem_t host_buf,
                                           const dev_mem_t dev_buf,
                                           size_t offset, size_t size)
{
    void *hptr = (void *) ((uintptr_t) host_buf.ptr + offset);
    void *dptr = (void *) ((uintptr_t) dev_buf.ptr + offset);
    cutilSafeCall(cudaMemcpyAsync(hptr, dptr, size,
                                  cudaMemcpyDeviceToHost, _stream));
    return 0;
}

void CUDAComputeContext::h2d_done(uint32_t task_id)
{
    return;
}

void CUDAComputeContext::d2h_done(uint32_t task_id)
{
    return;
}

void CUDAComputeContext::clear_kernel_args()
{
    num_kernel_args = 0;
}

void CUDAComputeContext::push_kernel_arg(struct kernel_arg &arg)
{
    assert(num_kernel_args < CUDA_MAX_KERNEL_ARGS);
    kernel_args[num_kernel_args ++] = arg;  /* Copied to the array. */
}

void CUDAComputeContext::push_common_kernel_args()
{
    struct kernel_arg arg = {(void *) &checkbits_d, sizeof(void *), alignof(void *)};
    this->push_kernel_arg(arg);
}

int CUDAComputeContext::enqueue_kernel_launch(dev_kernel_t kernel, struct resource_param *res)
{
    assert(checkbits_d != nullptr);
    // TODO: considerations for cudaFuncSetCacheConfig() and
    //       cudaSetDoubleFor*()?
    //cudaFuncAttributes attr;
    //cudaFuncGetAttributes(&attr, kernel.ptr);
    if (unlikely(res->num_workgroups == 0))
        res->num_workgroups = 1;

    num_workgroups = res->num_workgroups;

    void *raw_args[num_kernel_args];
    for (unsigned i = 0; i < num_kernel_args; i++) {
        raw_args[i] = kernel_args[i].ptr;
    }

    /* Clear checkbits. */
    unsigned n = (num_workgroups == 0) ? MAX_BLOCKS : num_workgroups;
    for (unsigned i = 0; i < num_workgroups; i++)
        checkbits_h[i] = 0;

    /* Launch! */
    state = ComputeContext::RUNNING;
    cutilSafeCall(cudaLaunchKernel(kernel.ptr, dim3(res->num_workgroups),
                                   dim3(res->num_threads_per_workgroup),
                                   (void **) &raw_args[0], 1024, _stream));
    return 0;
}

bool CUDAComputeContext::poll_input_finished(uint32_t task_id)
{
    /* Proceed to kernel launch without waiting. */
    return true;
}

bool CUDAComputeContext::poll_kernel_finished(uint32_t task_id)
{
    /* Check the checkbits if kernel has finished. */
    if (checkbits_h == nullptr) {
        return true;
    }
    for (unsigned i = 0; i < num_workgroups; i++) {
        if (checkbits_h[i] == 0) {
            return false;
        }
    }
    return true;
}

bool CUDAComputeContext::poll_output_finished(uint32_t task_id)
{
    cudaError_t ret = cudaStreamQuery(_stream);
    if (ret == cudaErrorNotReady)
        return false;
    // ignore non-cudaSuccess results...
    // (may happend on termination)
    return true;
}


int CUDAComputeContext::enqueue_event_callback(
        uint32_t task_id,
        void (*func_ptr)(ComputeContext *ctx, void *user_arg),
        void *user_arg)
{
    auto cb = [](cudaStream_t stream, cudaError_t status, void *user_data)
    {
        assert(status == cudaSuccess);
        struct cuda_event_context *cectx = (struct cuda_event_context *) user_data;
        cectx->callback(cectx->computectx, cectx->user_arg);
        delete cectx;
    };
    // TODO: how to avoid using new/delete?
    struct cuda_event_context *cectx = new struct cuda_event_context;
    cectx->computectx = this;
    cectx->callback = func_ptr;
    cectx->user_arg = user_arg;
    cutilSafeCall(cudaStreamAddCallback(_stream, cb, cectx, 0));
    return 0;
}


// vim: ts=8 sts=4 sw=4 et
