#ifndef __NBA_CUDA_COMPUTECTX_HH__
#define __NBA_CUDA_COMPUTECTX_HH__

#include <deque>

#include "../../lib/computedevice.hh"
#include "../../lib/computecontext.hh"
#include "../../lib/mempool.hh"
#include <cuda.h>
#include "mempool.hh"
#include "utils.hh"

namespace nba
{

class CUDAComputeContext: public ComputeContext
{
friend class CUDAComputeDevice;

private:
    CUDAComputeContext(unsigned ctx_id, ComputeDevice *mother_device);

public:
    virtual ~CUDAComputeContext();

    int alloc_input_buffer(size_t size, void **host_ptr, memory_t *dev_mem);
    int alloc_output_buffer(size_t size, void **host_ptr, memory_t *dev_mem);
    void clear_io_buffers();
    void *get_host_input_buffer_base();
    memory_t get_device_input_buffer_base();
    size_t get_total_input_buffer_size();
    void set_io_buffers(void *in_h, memory_t in_d, size_t in_sz,
                        void *out_h, memory_t out_d, size_t out_sz);
    void set_io_buffer_elemsizes(size_t *in_h, memory_t in_d, size_t in_sz,
                                 size_t *out_h, memory_t out_d, size_t out_sz);

    int enqueue_memwrite_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size);
    int enqueue_memread_op(void* host_buf, memory_t dev_buf, size_t offset, size_t size);
    int enqueue_kernel_launch(kernel_t kernel, struct resource_param *res,
                  struct kernel_arg *args, size_t num_args);
    int enqueue_event_callback(void (*func_ptr)(ComputeContext *ctx, void *user_arg), void *user_arg);

    cudaStream_t get_stream()
    {
        return _stream;
    }

    void sync()
    {
        cutilSafeCall(cudaStreamSynchronize(_stream));
    }

    bool query()
    {
        cudaError_t ret = cudaStreamQuery(_stream);
        if (ret == cudaErrorNotReady)
            return false;
        assert(ret == cudaSuccess);
        return true;
    }

    uint8_t *get_device_checkbits()
    {
        return checkbits_d;
    }

    uint8_t *get_host_checkbits()
    {
        return checkbits_h;
    }

    void clear_checkbits(unsigned num_workgroups = 0)
    {
        unsigned n = (num_workgroups == 0) ? MAX_BLOCKS : num_workgroups;
        for (unsigned i = 0; i < num_workgroups; i++)
            checkbits_h[i] = 0;
    }

    static const int MAX_BLOCKS = 16384;

private:
    uint8_t *checkbits_d;
    uint8_t *checkbits_h;
    cudaStream_t _stream;
    CUDAMemoryPool _cuda_mempool_in;
    CUDAMemoryPool _cuda_mempool_out;
    CPUMemoryPool _cpu_mempool_in;
    CPUMemoryPool _cpu_mempool_out;
};

}
#endif /*__NBA_CUDA_COMPUTECTX_HH__ */

// vim: ts=8 sts=4 sw=4 et
