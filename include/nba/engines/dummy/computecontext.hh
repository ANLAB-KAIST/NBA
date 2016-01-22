#ifndef __NBA_DUMMY_COMPUTECTX_HH__
#define __NBA_DUMMY_COMPUTECTX_HH__

#include <deque>

#include <nba/core/queue.hh>
#include <nba/core/mempool.hh>
#include <nba/framework/config.hh>
#include <nba/framework/computedevice.hh>
#include <nba/framework/computecontext.hh>
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

    io_base_t alloc_io_base();
    int alloc_input_buffer(io_base_t io_base, size_t size, void **host_ptr, memory_t *dev_mem);
    int alloc_output_buffer(io_base_t io_base, size_t size, void **host_ptr, memory_t *dev_mem);
    void get_input_current_pos(io_base_t io_base, void **host_ptr, memory_t *dev_mem) const;
    void get_output_current_pos(io_base_t io_base, void **host_ptr, memory_t *dev_mem) const;
    size_t get_input_size(io_base_t io_base) const;
    size_t get_output_size(io_base_t io_base) const;
    void clear_io_buffers(io_base_t io_base);

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
    DummyCPUMemoryPool _dev_mempool_in[NBA_MAX_IO_BASES];
    DummyCPUMemoryPool _dev_mempool_out[NBA_MAX_IO_BASES];
    DummyCPUMemoryPool _cpu_mempool_in[NBA_MAX_IO_BASES];
    DummyCPUMemoryPool _cpu_mempool_out[NBA_MAX_IO_BASES];

    FixedRing<unsigned> *io_base_ring;
};

}
#endif /*__NBA_DUMMY_COMPUTECTX_HH__ */

// vim: ts=8 sts=4 sw=4 et
