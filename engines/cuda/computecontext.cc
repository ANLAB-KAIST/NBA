#include "computecontext.hh"
#include "../../lib/log.hh"
#include "../../lib/common.hh"

using namespace std;
using namespace nba;

struct cuda_event_context {
    ComputeContext *computectx;
    void (*callback)(ComputeContext *ctx, void *user_arg);
    void *user_arg;
};

CUDAComputeContext::CUDAComputeContext(unsigned ctx_id, ComputeDevice *mother_device)
 : ComputeContext(ctx_id, mother_device), checkbits_d(NULL), checkbits_h(NULL),
   _cuda_mempool_in(), _cuda_mempool_out(),
   _cpu_mempool_in(cudaHostAllocPortable), _cpu_mempool_out(cudaHostAllocPortable)
   /* NOTE: Write-combined memory degrades performance to half... */
{
    type_name = "cuda";
    size_t mem_size = 8 * 1024 * 1024; // TODO: read from config
    cutilSafeCall(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
    _cuda_mempool_in.init(mem_size);
    _cuda_mempool_out.init(mem_size);
    _cpu_mempool_in.init(mem_size);
    _cpu_mempool_out.init(mem_size);
    cutilSafeCall(cudaHostAlloc((void **) &checkbits_h, MAX_BLOCKS, cudaHostAllocMapped));
    cutilSafeCall(cudaHostGetDevicePointer((void **) &checkbits_d, checkbits_h, 0));
    assert(checkbits_h != NULL);
    assert(checkbits_d != NULL);
    memset(checkbits_h, 0, MAX_BLOCKS);
}

CUDAComputeContext::~CUDAComputeContext()
{
    cutilSafeCall(cudaStreamDestroy(_stream));
    _cuda_mempool_in.destroy();
    _cuda_mempool_out.destroy();
    _cpu_mempool_in.destroy();
    _cpu_mempool_out.destroy();
    cutilSafeCall(cudaFreeHost(checkbits_h));
}

int CUDAComputeContext::alloc_input_buffer(size_t size, void **host_ptr, memory_t *dev_mem)
{
    *host_ptr = _cpu_mempool_in.alloc(size);
    assert(*host_ptr != nullptr);
    dev_mem->ptr = _cuda_mempool_in.alloc(size);
    assert(dev_mem->ptr != nullptr);
    return 0;
}

int CUDAComputeContext::alloc_output_buffer(size_t size, void **host_ptr, memory_t *dev_mem)
{
    *host_ptr = _cpu_mempool_out.alloc(size);
    assert(*host_ptr != nullptr);
    dev_mem->ptr = _cuda_mempool_out.alloc(size);
    assert(dev_mem->ptr != nullptr);
    return 0;
}

void CUDAComputeContext::clear_io_buffers()
{
    _cpu_mempool_in.reset();
    _cpu_mempool_out.reset();
    _cuda_mempool_in.reset();
    _cuda_mempool_out.reset();
}

void *CUDAComputeContext::get_host_input_buffer_base()
{
    return _cpu_mempool_in.get_base_ptr();
}

memory_t CUDAComputeContext::get_device_input_buffer_base()
{
    memory_t ret;
    ret.ptr = _cuda_mempool_in.get_base_ptr();
    return ret;
}

size_t CUDAComputeContext::get_total_input_buffer_size()
{
    assert(_cpu_mempool_in.get_alloc_size() == _cuda_mempool_in.get_alloc_size());
    return _cpu_mempool_in.get_alloc_size();
}

void CUDAComputeContext::set_io_buffers(void *in_h, memory_t in_d, size_t in_sz,
                    void *out_h, memory_t out_d, size_t out_sz)
{
    this->in_h = in_h;
    this->in_d = in_d;
    this->out_h = out_h;
    this->out_d = out_d;
    this->in_sz = in_sz;
    this->out_sz = out_sz;
}

void CUDAComputeContext::set_io_buffer_elemsizes(size_t *in_h, memory_t in_d, size_t in_sz,
                         size_t *out_h, memory_t out_d, size_t out_sz)
{
    this->in_elemsizes_h   = in_h;
    this->in_elemsizes_d   = in_d;
    this->out_elemsizes_h  = out_h;
    this->out_elemsizes_d  = out_d;
    this->in_elemsizes_sz  = in_sz;
    this->out_elemsizes_sz = out_sz;
}

int CUDAComputeContext::enqueue_memwrite_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    cutilSafeCall(cudaMemcpyAsync(dev_buf.ptr, host_buf, size, cudaMemcpyHostToDevice, _stream));
    return 0;
}

int CUDAComputeContext::enqueue_memread_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    cutilSafeCall(cudaMemcpyAsync(host_buf, dev_buf.ptr, size, cudaMemcpyDeviceToHost, _stream));
    return 0;
}

int CUDAComputeContext::enqueue_kernel_launch(kernel_t kernel, struct resource_param *res,
                          struct kernel_arg *args, size_t num_args)
{
    assert(checkbits_d != NULL);
    // TODO: considerations for cudaFuncSetCacheConfig() and
    //       cudaSetDoubleFor*()?
    //cudaFuncAttributes attr;
    //cudaFuncGetAttributes(&attr, kernel.ptr);
    // TODO: add extra room for dynamically allocated shared memory?
    if (unlikely(res->num_workgroups == 0))
        res->num_workgroups = 1;
    cutilSafeCall(cudaConfigureCall(res->num_workgroups, res->num_threads_per_workgroup,
                  1024, _stream));
                  //attr.sharedSizeBytes, _stream));
    size_t offset = 0;
    offset = ALIGN(offset, alignof(void*));
    cutilSafeCall(cudaSetupArgument(&in_d, sizeof(void*), offset));
    offset += sizeof(void*);
    offset = ALIGN(offset, alignof(void*));
    cutilSafeCall(cudaSetupArgument(&out_d, sizeof(void*), offset));
    offset += sizeof(void*);
    offset = ALIGN(offset, alignof(void*));
    cutilSafeCall(cudaSetupArgument(&in_elemsizes_d, sizeof(void*), offset));
    offset += sizeof(void*);
    offset = ALIGN(offset, alignof(void*));
    cutilSafeCall(cudaSetupArgument(&out_elemsizes_d, sizeof(void*), offset));
    offset += sizeof(void*);
    offset = ALIGN(offset, alignof(uint32_t));
    cutilSafeCall(cudaSetupArgument(&res->num_workitems, sizeof(uint32_t), offset));
    offset += sizeof(uint32_t);
    offset = ALIGN(offset, alignof(uint8_t*));
    cutilSafeCall(cudaSetupArgument(&checkbits_d, sizeof(uint8_t*), offset));
    offset += sizeof(uint8_t*);
    for (unsigned i = 0; i < num_args; i++) {
        offset = ALIGN(offset, args[i].align);
        cutilSafeCall(cudaSetupArgument(args[i].ptr, args[i].size, offset));
        offset += args[i].size;
    }
    state = ComputeContext::RUNNING;
    cutilSafeCall(cudaLaunch(kernel.ptr));
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
