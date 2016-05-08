#ifndef __CUDA_ENGINE_HH__
#define __CUDA_ENGINE_HH__

#include <string>
#include <vector>
#include <deque>

#include <nba/framework/computedevice.hh>
#include <nba/core/threading.hh>
#include <cuda.h>

namespace nba
{

class CUDAComputeContext;

class CUDAComputeDevice: public ComputeDevice
{
public:
    friend class CUDAComputeContext;

    CUDAComputeDevice(unsigned node_id, unsigned device_id, size_t num_contexts);
    virtual ~CUDAComputeDevice();

    int get_spec(struct compute_device_spec *spec);
    int get_utilization(struct compute_device_util *util);
    host_mem_t alloc_host_buffer(size_t size, int flags);
    dev_mem_t alloc_device_buffer(size_t size, int flags, host_mem_t &assoc_host_buf);
    void free_host_buffer(host_mem_t m);
    void free_device_buffer(dev_mem_t m);
    void *unwrap_host_buffer(const host_mem_t m);
    void *unwrap_device_buffer(const dev_mem_t m);
    void memwrite(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size);
    void memread(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size);

private:
    ComputeContext *_get_available_context();
    void _return_context(ComputeContext *ctx);

    std::deque<CUDAComputeContext *> _ready_contexts;
    std::deque<CUDAComputeContext *> _active_contexts;
    Lock _lock;
    CondVar _ready_cond;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
