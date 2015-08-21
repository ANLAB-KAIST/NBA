#ifndef __PHI_ENGINE_HH__
#define __PHI_ENGINE_HH__

#include <string>
#include <vector>
#include <deque>
#include <CL/opencl.h>

#include <nba/framework/computedevice.hh>
#include <nba/framework/computecontext.hh>
#include <nba/core/threading.hh>
#include <nba/engines/phi/utils.hh>
#include <nba/engines/phi/computecontext.hh>

namespace nba
{

class PhiComputeContext;

class PhiComputeDevice: public ComputeDevice
{
public:
    friend class PhiComputeContext;

    PhiComputeDevice(unsigned node_id, unsigned device_id, size_t num_contexts);
    virtual ~PhiComputeDevice();

    int get_spec(struct compute_device_spec *spec);
    int get_utilization(struct compute_device_util *util);
    void *alloc_host_buffer(size_t size, int flags);
    memory_t alloc_device_buffer(size_t size, int flags);
    void free_host_buffer(void *ptr);
    void free_device_buffer(memory_t ptr);
    void memwrite(void *host_buf, memory_t dev_buf, size_t offset, size_t size);
    void memread(void *host_buf, memory_t dev_buf, size_t offset, size_t size);

private:
    cl_context clctx;
    cl_command_queue cldefqueue;
    cl_device_id cldevid;
    ComputeContext *_get_available_context();
    void _return_context(ComputeContext *ctx);

    std::deque<PhiComputeContext *> _ready_contexts;
    std::deque<PhiComputeContext *> _active_contexts;
    Lock _lock;
    CondVar _ready_cond;
};

}

#endif /* __NBA_PHI_ENGINE_HH__ */

// vim: ts=8 sts=4 sw=4 et
