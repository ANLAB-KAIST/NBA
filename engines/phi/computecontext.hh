#ifndef __NSHADER_PHI_COMPUTECTX_HH__
#define __NSHADER_PHI_COMPUTECTX_HH__

#include <deque>
#include <CL/opencl.h>

#include "../../lib/computedevice.hh"
#include "../../lib/computecontext.hh"
#include "utils.hh"
#include "mempool.hh"

namespace nshader
{

#define PHI_MAX_KERNEL_ARGS     (16)

class PhiComputeContext: public ComputeContext
{
friend class PhiComputeDevice;

private:
    PhiComputeContext(unsigned ctx_id, ComputeDevice *mother_device);

public:
    virtual ~PhiComputeContext();

    int alloc_input_buffer(size_t size, void **host_ptr, memory_t *dev_mem);
    int alloc_output_buffer(size_t size, void **host_ptr, memory_t *dev_mem);
    void clear_io_buffers();
    void *get_host_input_buffer_base();
    memory_t get_device_input_buffer_base();
    size_t get_total_input_buffer_size();

    void clear_kernel_args() { }
    void push_kernel_arg(struct kernel_arg &arg) { }

    int enqueue_memwrite_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size);
    int enqueue_memread_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size);
    int enqueue_kernel_launch(kernel_t kernel, struct resource_param *res);
    int enqueue_event_callback(void (*func_ptr)(ComputeContext *ctx, void *user_arg), void *user_arg);

    void *get_stream()
    {
        // TODO: implement
        return NULL;
    }
    //cudaStream_t get_stream()
    //{
    //  return _stream;
    //}

    void sync()
    {
        clFinish(clqueue);
    }

    bool query()
    {
        /* Check the "last" issued event.
         * Here is NOT a synchronization point. */
        cl_int status;
        if (clev == nullptr) // No async commands are issued.
            return true;
        phiSafeCall(clGetEventInfo(clev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                   sizeof(cl_int), &status, NULL));
        return (status == CL_COMPLETE);
    }

    uint8_t *get_device_checkbits()
    {
        assert(false, "not implemented");
        return nullptr;
    }

    uint8_t *get_host_checkbits()
    {
        return checkbits_h;
    }

    void clear_checkbits(unsigned num_workgroups)
    {
        unsigned n = (num_workgroups == 0) ? MAX_BLOCKS : num_workgroups;
        for (unsigned i = 0; i < num_workgroups; i++)
            checkbits_h[i] = 0;
    }

    static const int MAX_BLOCKS = 16384;

private:
    memory_t checkbits_d;
    uint8_t *checkbits_h;
    cl_command_queue clqueue;
    cl_event clev;
    cl_event clev_marker;
    PhiMemoryPool *dev_mempool_in;
    PhiMemoryPool *dev_mempool_out;
    CPUMemoryPool *cpu_mempool_in;
    CPUMemoryPool *cpu_mempool_out;

    size_t num_kernel_args;
    struct kernel_arg kernel_args[CUDA_MAX_KERNEL_ARGS];
};

}
#endif /* __NSHADER_PHI_COMPUTECTX_HH__ */

// vim: ts=8 sts=4 sw=4 et
