#include <nba/core/intrinsic.hh>
#include <nba/framework/logging.hh>
#include <nba/engines/cuda/computedevice.hh>

using namespace std;
using namespace nba;

CUDAComputeDevice::CUDAComputeDevice(
        unsigned node_id, unsigned device_id, size_t num_contexts
) : ComputeDevice(node_id, device_id, num_contexts)
{
    type_name = "cuda";
    assert(num_contexts > 0);
    cutilSafeCall(cudaSetDevice(device_id));
    cutilSafeCall(cudaSetDeviceFlags(cudaDeviceScheduleSpin & cudaDeviceMapHost));
    cutilSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    RTE_LOG(DEBUG, COPROC, "CUDAComputeDevice: # contexts: %lu\n", num_contexts);
    for (unsigned i = 0; i < num_contexts; i++) {
        CUDAComputeContext *ctx = new CUDAComputeContext(i, this);
        _ready_contexts.push_back(ctx);
        contexts.push_back((ComputeContext *) ctx);
    }
}

CUDAComputeDevice::~CUDAComputeDevice()
{
    for (auto it = _ready_contexts.begin(); it != _ready_contexts.end(); it++) {
        CUDAComputeContext *ctx = *it;
        delete ctx;
        *it = NULL;
    }
    for (auto it = _active_contexts.begin(); it != _active_contexts.end(); it++) {
        CUDAComputeContext *ctx = *it;
        delete ctx;
        *it = NULL;
    }
}

int CUDAComputeDevice::get_spec(struct compute_device_spec *spec)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    spec->max_threads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    spec->max_workgroups = prop.maxGridSize[0] * prop.maxGridSize[1] * prop.maxGridSize[2];
    spec->max_concurrent_kernels = prop.concurrentKernels;
    spec->global_memory_size = prop.totalGlobalMem;
    return 0;
}

int CUDAComputeDevice::get_utilization(struct compute_device_util *util)
{
    size_t free, total;
    cutilSafeCall(cudaMemGetInfo(&free, &total));
    util->used_memory_bytes = total - free;
    // TODO: true utilization value can be read from NVML library,
    //       but our GeForce-class GPUs do not support it. :(
    //       Any better estimation??
    util->utilization = (float)free / total;
    return 0;
}

ComputeContext *CUDAComputeDevice::_get_available_context()
{
    _ready_cond.lock();
    CUDAComputeContext *cctx = _ready_contexts.front();
    assert(cctx != NULL);
    _ready_contexts.pop_front();
    _active_contexts.push_back(cctx);
    _ready_cond.unlock();
    return (ComputeContext *) cctx;
}

void CUDAComputeDevice::_return_context(ComputeContext *cctx)
{
    /* This method is called inside CUDA's own thread. */
    assert(cctx != NULL);
    /* We do linear search here, it would not be a big overhead since
     * the number of contexts are small (less than 16 for CUDA). */
    _ready_cond.lock();
    assert(_ready_contexts.size() < num_contexts);
    for (auto it = _active_contexts.begin(); it != _active_contexts.end(); it++) {
        if (cctx == *it) {
            _active_contexts.erase(it);
            _ready_contexts.push_back((CUDAComputeContext *) cctx);
            break;
        }
    }
    _ready_cond.unlock();
}

host_mem_t CUDAComputeDevice::alloc_host_buffer(size_t size, int flags)
{
    void *ptr;
    int nvflags = 0;
    nvflags |= (flags & HOST_PINNED) ? cudaHostAllocPortable : 0;
    nvflags |= (flags & HOST_MAPPED) ? cudaHostAllocMapped : 0;
    nvflags |= (flags & HOST_WRITECOMBINED) ? cudaHostAllocWriteCombined : 0;
    cutilSafeCall(cudaHostAlloc(&ptr, size, nvflags));
    assert(ptr != NULL);
    return { ptr };
}

dev_mem_t CUDAComputeDevice::alloc_device_buffer(size_t size, int flags)
{
    void *ptr;
    cutilSafeCall(cudaMalloc(&ptr, size));
    assert(ptr != NULL);
    return { ptr };
}

void CUDAComputeDevice::free_host_buffer(host_mem_t m)
{
    cutilSafeCall(cudaFreeHost(m.ptr));
}

void CUDAComputeDevice::free_device_buffer(dev_mem_t m)
{
    cutilSafeCall(cudaFree(m.ptr));
}

void CUDAComputeDevice::memwrite(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    cutilSafeCall(cudaMemcpy((uint8_t *) dev_buf.ptr + offset, host_buf.ptr, size, cudaMemcpyHostToDevice));
}

void CUDAComputeDevice::memread(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    cutilSafeCall(cudaMemcpy(host_buf.ptr, (uint8_t *) dev_buf.ptr + offset, size, cudaMemcpyDeviceToHost));
}

// vim: ts=8 sts=4 sw=4 et
