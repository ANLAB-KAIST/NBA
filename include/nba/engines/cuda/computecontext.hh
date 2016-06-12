#ifndef __NBA_CUDA_COMPUTECTX_HH__
#define __NBA_CUDA_COMPUTECTX_HH__

#include <nba/core/queue.hh>
#include <nba/framework/config.hh>
#include <nba/framework/computedevice.hh>
#include <nba/framework/computecontext.hh>
#include <cuda.h>
#include <nba/engines/cuda/utils.hh>

struct rte_memzone;

#define CUDA_MAX_KERNEL_ARGS    (16)

namespace nba
{

class CUDAMemoryPool;
class CPUMemoryPool;

class CUDAComputeContext: public ComputeContext
{
friend class CUDAComputeDevice;

private:
    CUDAComputeContext(unsigned ctx_id, ComputeDevice *mother_device);

public:
    virtual ~CUDAComputeContext();

    uint32_t alloc_task_id();
    void release_task_id(uint32_t task_id);
    io_base_t alloc_io_base();
    int alloc_input_buffer(io_base_t io_base, size_t size,
                           host_mem_t &host_ptr, dev_mem_t &dev_ptr);
    int alloc_inout_buffer(io_base_t io_base, size_t size,
                           host_mem_t &host_ptr, dev_mem_t &dev_ptr);
    int alloc_output_buffer(io_base_t io_base, size_t size,
                            host_mem_t &host_ptr, dev_mem_t &dev_ptr);
    void get_input_buffer(io_base_t io_base,
                          host_mem_t &hbuf, dev_mem_t &dbuf) const;
    void get_inout_buffer(io_base_t io_base,
                          host_mem_t &hbuf, dev_mem_t &dbuf) const;
    void get_output_buffer(io_base_t io_base,
                           host_mem_t &hbuf, dev_mem_t &dbuf) const;
    void *unwrap_host_buffer(const host_mem_t hbuf) const;
    void *unwrap_device_buffer(const dev_mem_t dbuf) const;
    size_t get_input_size(io_base_t io_base) const;
    size_t get_inout_size(io_base_t io_base) const;
    size_t get_output_size(io_base_t io_base) const;
    void shift_inout_base(io_base_t io_base, size_t len);
    void clear_io_buffers(io_base_t io_base);

    void clear_kernel_args();
    void push_kernel_arg(struct kernel_arg &arg);
    void push_common_kernel_args();

    int enqueue_memwrite_op(uint32_t task_id,
                            const host_mem_t host_buf, const dev_mem_t dev_buf,
                            size_t offset, size_t size);
    int enqueue_memread_op(uint32_t task_id,
                           const host_mem_t host_buf, const dev_mem_t dev_buf,
                           size_t offset, size_t size);
    int enqueue_kernel_launch(dev_kernel_t kernel, struct resource_param *res);
    int enqueue_event_callback(uint32_t task_id,
                               void (*func_ptr)(ComputeContext *ctx, void *user_arg),
                               void *user_arg);

    void h2d_done(uint32_t task_id);
    void d2h_done(uint32_t task_id);
    bool poll_input_finished(uint32_t task_id);
    bool poll_kernel_finished(uint32_t task_id);
    bool poll_output_finished(uint32_t task_id);

    void sync()
    {
        cutilSafeCall(cudaStreamSynchronize(_stream));
    }

    static const int MAX_BLOCKS = 16384;

private:

    uint8_t *checkbits_d;
    uint8_t *checkbits_h;
    uint32_t num_workgroups;
    cudaStream_t _stream;
    CUDAMemoryPool *_cuda_mempool_in[NBA_MAX_IO_BASES];
    CPUMemoryPool  *_cuda_mempool_inout[NBA_MAX_IO_BASES];
    CUDAMemoryPool *_cuda_mempool_out[NBA_MAX_IO_BASES];
    CPUMemoryPool *_cpu_mempool_in[NBA_MAX_IO_BASES];
    CPUMemoryPool *_cpu_mempool_inout[NBA_MAX_IO_BASES];
    CPUMemoryPool *_cpu_mempool_out[NBA_MAX_IO_BASES];

    const struct rte_memzone *reserve_memory(ComputeDevice *mother);
    const struct rte_memzone *mz;

    size_t num_kernel_args;
    struct kernel_arg kernel_args[CUDA_MAX_KERNEL_ARGS];

    FixedRing<unsigned> *io_base_ring;
    uint32_t next_task_id;
};

}
#endif /*__NBA_CUDA_COMPUTECTX_HH__ */

// vim: ts=8 sts=4 sw=4 et
