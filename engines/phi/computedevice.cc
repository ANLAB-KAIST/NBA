#include "computedevice.hh"
#include "../../lib/log.hh"

using namespace std;
using namespace nshader;

PhiComputeDevice::PhiComputeDevice(
        unsigned node_id, unsigned device_id, size_t num_contexts
) : ComputeDevice(node_id, device_id, num_contexts)
{
    type_name = "cuda";
    assert(num_contexts > 0);

    cl_int err_ret;
    cldevid = (cl_device_id) device_id;
    clctx = clCreateContext(NULL, 1, &cldevid, NULL, NULL, &err_ret);
    if (err_ret != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext()@PhiComputeDevice() failed.\n");
        exit(1);
    }
    /* Create a "default" command queue for synchronous operations. */
    cldefqueue = clCreateCommandQueue(clctx, cldevid, 0, &err_ret);
    if (err_ret != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueue()@PhiComputeDevice() failed.\n");
    }
    RTE_LOG(DEBUG, COPROC, "PhiComputeDevice: # contexts: %lu\n", num_contexts);
    for (unsigned i = 0; i < num_contexts; i++) {
        PhiComputeContext *ctx = new PhiComputeContext(i, this);
        _ready_contexts.push_back(ctx);
        contexts.push_back((ComputeContext *) ctx);
    }
}

PhiComputeDevice::~PhiComputeDevice()
{
    for (auto it = _ready_contexts.begin(); it != _ready_contexts.end(); it++) {
        PhiComputeContext *ctx = *it;
        delete ctx;
        *it = NULL;
    }
    for (auto it = _active_contexts.begin(); it != _active_contexts.end(); it++) {
        PhiComputeContext *ctx = *it;
        delete ctx;
        *it = NULL;
    }
    clReleaseCommandQueue(cldefqueue);
    clReleaseContext(clctx);
}

int PhiComputeDevice::get_spec(struct compute_device_spec *spec)
{
    //cudaDeviceProp prop;
    //cudaGetDeviceProperties(&prop, device_id);
    //spec->max_threads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    //spec->max_workgroups = prop.maxGridSize[0] * prop.maxGridSize[1] * prop.maxGridSize[2];
    //spec->max_concurrent_kernels = prop.concurrentKernels;
    //spec->global_memory_size = prop.totalGlobalMem;
    // TODO: implement
    return 0;
}

int PhiComputeDevice::get_utilization(struct compute_device_util *util)
{
    size_t free = 0, total = 1;
    //phiSafeCall(cudaMemGetInfo(&free, &total));
    util->used_memory_bytes = total - free;
    // TODO: true utilization value can be read from NVML library,
    //       but our GeForce-class GPUs do not support it. :(
    //       Any better estimation??
    util->utilization = (float)free / total;
    return 0;
}

ComputeContext *PhiComputeDevice::_get_available_context()
{
    _ready_cond.lock();
    PhiComputeContext *cctx = _ready_contexts.front();
    assert(cctx != NULL);
    _ready_contexts.pop_front();
    _active_contexts.push_back(cctx);
    _ready_cond.unlock();
    return (ComputeContext *) cctx;
}

void PhiComputeDevice::_return_context(ComputeContext *cctx)
{
    /* This method is called inside Phi's own thread. */
    assert(cctx != NULL);
    /* We do linear search here, it would not be a big overhead since
     * the number of contexts are small (less than 16 for Phi). */
    _ready_cond.lock();
    assert(_ready_contexts.size() < num_contexts);
    for (auto it = _active_contexts.begin(); it != _active_contexts.end(); it++) {
        if (cctx == *it) {
            _active_contexts.erase(it);
            _ready_contexts.push_back((PhiComputeContext *) cctx);
            break;
        }
    }
    _ready_cond.unlock();
}

void *PhiComputeDevice::alloc_host_buffer(size_t size, int flags)
{
    return malloc(size);
}

memory_t PhiComputeDevice::alloc_device_buffer(size_t size, int flags)
{
    memory_t ret;
    cl_int err_ret;
    ret.clmem = clCreateBuffer(clctx, CL_MEM_READ_WRITE, size, NULL, &err_ret);
    // TODO: implement details for various flags
    if (err_ret != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer()@PhiComputeDevice::alloc_device_buffer() failed\n");
        ret.ptr = NULL;
        return ret;
    }
    return ret;
}

void PhiComputeDevice::free_host_buffer(void *m)
{
    free(m);
}

void PhiComputeDevice::free_device_buffer(memory_t m)
{
    clReleaseMemObject(m.clmem);
}

void PhiComputeDevice::memwrite(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    clEnqueueWriteBuffer(cldefqueue, dev_buf.clmem, CL_TRUE, offset, size, host_buf, 0, NULL, NULL);
}

void PhiComputeDevice::memread(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    clEnqueueReadBuffer(cldefqueue, dev_buf.clmem, CL_TRUE, offset, size, host_buf, 0, NULL, NULL);
}

// vim: ts=8 sts=4 sw=4 et
