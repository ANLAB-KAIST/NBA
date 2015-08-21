#ifndef __NBA_COMPUTECONTEXT_HH__
#define __NBA_COMPUTECONTEXT_HH__

#include <nba/core/intrinsic.hh>
#include <nba/core/offloadtypes.hh>
#include <nba/framework/computedevice.hh>

namespace nba {

class OffloadTask;

class ComputeContext {

    /**
     * ComputeContext represents a command queue, such as CUDA streams.
     */
public:
    static const unsigned READY = 0;
    static const unsigned PREPARING = 1;
    static const unsigned RUNNING = 2;
    static const unsigned FINISHED = 3;
    static const unsigned PTR_STORAGE_SIZE = 16;

    ComputeContext(unsigned ctx_id, ComputeDevice *mother_device)
                   : type_name("<invalid>"), state(READY), currently_running_task(NULL),
                     mother_device(mother_device),
                     node_id(mother_device->node_id),
                     device_id(mother_device->device_id),
                     ctx_id(ctx_id)
    {
    }
    virtual ~ComputeContext() {}

    virtual int alloc_input_buffer(size_t size, void **host_ptr, memory_t *dev_mem) = 0;
    virtual int alloc_output_buffer(size_t size, void **host_ptr, memory_t *dev_mem) = 0;
    virtual void clear_io_buffers() = 0;
    virtual void *get_host_input_buffer_base() = 0;
    virtual memory_t get_device_input_buffer_base() = 0;
    virtual size_t get_total_input_buffer_size() = 0;

    virtual void clear_kernel_args() = 0;
    virtual void push_kernel_arg(struct kernel_arg &arg) = 0;

    /* All functions must be implemented as non-blocking. */
    virtual int enqueue_memwrite_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size) = 0;
    virtual int enqueue_memread_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size) = 0;
    virtual int enqueue_kernel_launch(kernel_t kernel, struct resource_param *res) = 0;
    virtual int enqueue_event_callback(void (*func_ptr)(ComputeContext *ctx, void *user_arg), void *user_arg) = 0;

    virtual uint8_t *get_device_checkbits() = 0;
    virtual uint8_t *get_host_checkbits() = 0;
    virtual void clear_checkbits(unsigned num_workgroups = 0) = 0;
    virtual void sync() = 0;
    virtual bool query() = 0;
    unsigned get_id()
    {
        return ctx_id;
    }

public:
    std::string type_name;
    volatile unsigned state __cache_aligned;
    OffloadTask *currently_running_task;

protected:
    const ComputeDevice *mother_device;
    const unsigned node_id;
    const unsigned device_id;
    const unsigned ctx_id;
};

}
#endif

// vim: ts=8 sts=4 sw=4 et
