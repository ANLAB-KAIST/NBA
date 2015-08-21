#ifndef __NBA_DUMMY_COMPUTECTX_HH__
#define __NBA_DUMMY_COMPUTECTX_HH__

#include <deque>

#include <nba/framework/computedevice.hh>
#include <nba/framework/computecontext.hh>
#include <nba/core/mempool.hh>
#include <nba/engines/dummy/mempool.hh>

namespace nba
{

class DummyComputeContext: public ComputeContext
{
friend class DummyComputeDevice;

private:
    DummyComputeContext(unsigned ctx_id, ComputeDevice *mother_device);

public:
    virtual ~DummyComputeContext();

    int alloc_input_buffer(size_t size, void **host_ptr, memory_t *dev_mem);
    int alloc_output_buffer(size_t size, void **host_ptr, memory_t *dev_mem);
    void clear_io_buffers();
    void *get_host_input_buffer_base();
    memory_t get_device_input_buffer_base();
    size_t get_total_input_buffer_size();

    void clear_kernel_args() { }
    void push_kernel_arg(struct kernel_arg &arg) { }

    int enqueue_memwrite_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size);
    int enqueue_memread_op(void* host_buf, memory_t dev_buf, size_t offset, size_t size);
    int enqueue_kernel_launch(kernel_t kernel, struct resource_param *res);
    int enqueue_event_callback(void (*func_ptr)(ComputeContext *ctx, void *user_arg), void *user_arg);

    void sync()
    {
        return;
    }

    bool query()
    {
        return true;
    }

    uint8_t *get_device_checkbits()
    {
        return nullptr;
    }

    uint8_t *get_host_checkbits()
    {
        return nullptr;
    }

    void clear_checkbits(unsigned num_workgroups)
    {
        return;
    }

private:
    DummyCPUMemoryPool _dev_mempool_in;
    DummyCPUMemoryPool _dev_mempool_out;
    DummyCPUMemoryPool _cpu_mempool_in;
    DummyCPUMemoryPool _cpu_mempool_out;
};

}
#endif /*__NBA_DUMMY_COMPUTECTX_HH__ */

// vim: ts=8 sts=4 sw=4 et
