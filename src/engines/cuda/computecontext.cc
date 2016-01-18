#include <nba/core/intrinsic.hh>
#include <nba/engines/cuda/computecontext.hh>
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
#undef USE_PHYS_CONT_MEMORY // performance degraded :(

CUDAComputeContext::CUDAComputeContext(unsigned ctx_id, ComputeDevice *mother)
 : ComputeContext(ctx_id, mother), checkbits_d(NULL), checkbits_h(NULL),
   mz(reserve_memory(mother)), num_kernel_args(0)
   /* NOTE: Write-combined memory degrades performance to half... */
{
    type_name = "cuda";
    size_t io_base_size = ALIGN_CEIL(IO_BASE_SIZE, getpagesize()); // TODO: read from config
    cutilSafeCall(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
    io_base_ring.init(NBA_MAX_IO_BASES, node_id, io_base_ring_buf);
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        io_base_ring.push_back(i);
        _cuda_mempool_in[i].init(io_base_size);
        _cuda_mempool_out[i].init(io_base_size);
        #ifdef USE_PHYS_CONT_MEMORY
        void *base;
        base = (void *) ((uintptr_t) mz->addr + i * io_base_size);
        _cpu_mempool_in[i].init_with_flags(io_base_size, base, 0);
        base = (void *) ((uintptr_t) mz->addr + i * io_base_size + NBA_MAX_IO_BASES * io_base_size);
        _cpu_mempool_out[i].init_with_flags(io_base_size, base, 0);
        #else
        _cpu_mempool_in[i].init_with_flags(io_base_size, nullptr, cudaHostAllocPortable);
        _cpu_mempool_out[i].init_with_flags(io_base_size, nullptr, cudaHostAllocPortable);
        #endif
    }
    {
        void *t;
        cutilSafeCall(cudaMalloc((void **) &t, CACHE_LINE_SIZE));
        dummy_dev_buf.ptr = t;
        cutilSafeCall(cudaHostAlloc((void **) &t, CACHE_LINE_SIZE, cudaHostAllocPortable));
        dummy_host_buf = t;
    }
    cutilSafeCall(cudaHostAlloc((void **) &checkbits_h, MAX_BLOCKS, cudaHostAllocMapped));
    cutilSafeCall(cudaHostGetDevicePointer((void **) &checkbits_d, checkbits_h, 0));
    assert(checkbits_h != NULL);
    assert(checkbits_d != NULL);
    memset(checkbits_h, 0, MAX_BLOCKS);
}

const struct rte_memzone *CUDAComputeContext::reserve_memory(ComputeDevice *mother)
{
#ifdef USE_PHYS_CONT_MEMORY
    char namebuf[RTE_MEMZONE_NAMESIZE];
    size_t io_base_size = ALIGN_CEIL(IO_BASE_SIZE, getpagesize());
    snprintf(namebuf, RTE_MEMZONE_NAMESIZE, "cuda.io.%d:%d", mother->device_id, ctx_id);
    const struct rte_memzone *_mz = rte_memzone_reserve(namebuf, 2 * io_base_size * NBA_MAX_IO_BASES,
                                                        mother->node_id,
                                                        RTE_MEMZONE_2MB | RTE_MEMZONE_SIZE_HINT_ONLY);
    assert(_mz != nullptr);
    return _mz;
#else
    return nullptr;
#endif
}

CUDAComputeContext::~CUDAComputeContext()
{
    cutilSafeCall(cudaStreamDestroy(_stream));
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        _cuda_mempool_in[i].destroy();
        _cuda_mempool_out[i].destroy();
        _cpu_mempool_in[i].destroy();
        _cpu_mempool_out[i].destroy();
    }
    if (mz != nullptr)
        rte_memzone_free(mz);
    cutilSafeCall(cudaFreeHost(checkbits_h));
}

io_base_t CUDAComputeContext::alloc_io_base()
{
    if (io_base_ring.empty()) return INVALID_IO_BASE;
    unsigned i = io_base_ring.front();
    io_base_ring.pop_front();
    return (io_base_t) i;
}

void CUDAComputeContext::get_input_current_pos(io_base_t io_base, void **host_ptr, memory_t *dev_mem) const
{
    unsigned i = io_base;
    *host_ptr = (char*)_cpu_mempool_in[i].get_base_ptr() + (uintptr_t)_cpu_mempool_in[i].get_alloc_size();
    dev_mem->ptr = (char*)_cuda_mempool_in[i].get_base_ptr() + (uintptr_t)_cuda_mempool_in[i].get_alloc_size();
}

void CUDAComputeContext::get_output_current_pos(io_base_t io_base, void **host_ptr, memory_t *dev_mem) const
{
    unsigned i = io_base;
    *host_ptr = (char*)_cpu_mempool_out[i].get_base_ptr() + (uintptr_t)_cpu_mempool_out[i].get_alloc_size();
    dev_mem->ptr = (char*)_cuda_mempool_out[i].get_base_ptr() + (uintptr_t)_cuda_mempool_out[i].get_alloc_size();
}

size_t CUDAComputeContext::get_input_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_in[i].get_alloc_size();
}

size_t CUDAComputeContext::get_output_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_out[i].get_alloc_size();
}

int CUDAComputeContext::alloc_input_buffer(io_base_t io_base, size_t size, void **host_ptr, memory_t *dev_mem)
{
    unsigned i = io_base;
    *host_ptr = _cpu_mempool_in[i].alloc(size);
    assert(*host_ptr != nullptr);
    dev_mem->ptr = _cuda_mempool_in[i].alloc(size);
    assert(dev_mem->ptr != nullptr);
    return 0;
}

int CUDAComputeContext::alloc_output_buffer(io_base_t io_base, size_t size, void **host_ptr, memory_t *dev_mem)
{
    unsigned i = io_base;
    *host_ptr = _cpu_mempool_out[i].alloc(size);
    assert(*host_ptr != nullptr);
    dev_mem->ptr = _cuda_mempool_out[i].alloc(size);
    assert(dev_mem->ptr != nullptr);
    return 0;
}

void CUDAComputeContext::clear_io_buffers(io_base_t io_base)
{
    unsigned i = io_base;
    _cpu_mempool_in[i].reset();
    _cpu_mempool_out[i].reset();
    _cuda_mempool_in[i].reset();
    _cuda_mempool_out[i].reset();
    io_base_ring.push_back(i);
}

int CUDAComputeContext::enqueue_memwrite_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    //cutilSafeCall(cudaMemcpyAsync(dummy_dev_buf.ptr, dummy_host_buf, 64, cudaMemcpyHostToDevice, _stream));
    cutilSafeCall(cudaMemcpyAsync(dev_buf.ptr, host_buf, size, cudaMemcpyHostToDevice, _stream));
    return 0;
}

int CUDAComputeContext::enqueue_memread_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    //cutilSafeCall(cudaMemcpyAsync(dummy_host_buf, dummy_dev_buf.ptr, 64, cudaMemcpyDeviceToHost, _stream));
    cutilSafeCall(cudaMemcpyAsync(host_buf, dev_buf.ptr, size, cudaMemcpyDeviceToHost, _stream));
    return 0;
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

int CUDAComputeContext::enqueue_kernel_launch(kernel_t kernel, struct resource_param *res)
{
    assert(checkbits_d != nullptr);
    // TODO: considerations for cudaFuncSetCacheConfig() and
    //       cudaSetDoubleFor*()?
    //cudaFuncAttributes attr;
    //cudaFuncGetAttributes(&attr, kernel.ptr);
    if (unlikely(res->num_workgroups == 0))
        res->num_workgroups = 1;
    void *raw_args[num_kernel_args];
    for (unsigned i = 0; i < num_kernel_args; i++) {
        raw_args[i] = kernel_args[i].ptr;
    }
    state = ComputeContext::RUNNING;
    cutilSafeCall(cudaLaunchKernel(kernel.ptr, dim3(res->num_workgroups),
                                   dim3(res->num_threads_per_workgroup),
                                   (void **) &raw_args[0], 1024, _stream));
    return 0;
}

int CUDAComputeContext::enqueue_event_callback(void (*func_ptr)(ComputeContext *ctx, void *user_arg), void *user_arg)
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
