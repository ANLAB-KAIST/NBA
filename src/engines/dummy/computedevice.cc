#include <nba/framework/logging.hh>
#include <nba/engines/dummy/computedevice.hh>

using namespace std;
using namespace nba;

DummyComputeDevice::DummyComputeDevice(
        unsigned node_id, unsigned device_id, size_t num_contexts
) : ComputeDevice(node_id, device_id, num_contexts)
{
    type_name = "dummy";
    assert(num_contexts > 0);
    RTE_LOG(DEBUG, COPROC, "DummyComputeDevice: # contexts: %lu\n", num_contexts);
    for (unsigned i = 0; i < num_contexts; i++) {
        DummyComputeContext *ctx = new DummyComputeContext(i, this);
        _ready_contexts.push_back(ctx);
        contexts.push_back((ComputeContext *) ctx);
    }
}

DummyComputeDevice::~DummyComputeDevice()
{
    for (auto it = _ready_contexts.begin(); it != _ready_contexts.end(); it++) {
        DummyComputeContext *ctx = *it;
        delete ctx;
        *it = NULL;
    }
    for (auto it = _active_contexts.begin(); it != _active_contexts.end(); it++) {
        DummyComputeContext *ctx = *it;
        delete ctx;
        *it = NULL;
    }
}

int DummyComputeDevice::get_spec(struct compute_device_spec *spec)
{
    spec->max_threads = 1;
    spec->max_workgroups = 1;
    spec->max_concurrent_kernels = 16;
    spec->global_memory_size = 1024 * 1024 * 1024;
    return 0;
}

int DummyComputeDevice::get_utilization(struct compute_device_util *util)
{
    util->used_memory_bytes = 0;
    util->utilization = 0.5f;
    return 0;
}

ComputeContext *DummyComputeDevice::_get_available_context()
{
    _ready_cond.lock();
    DummyComputeContext *cctx = _ready_contexts.front();
    assert(cctx != NULL);
    _ready_contexts.pop_front();
    _active_contexts.push_back(cctx);
    _ready_cond.unlock();
    return (ComputeContext *) cctx;
}

void DummyComputeDevice::_return_context(ComputeContext *cctx)
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
            _ready_contexts.push_back((DummyComputeContext *) cctx);
            break;
        }
    }
    _ready_cond.unlock();
}

host_mem_t DummyComputeDevice::alloc_host_buffer(size_t size, int flags)
{
    return { malloc(size) };
}

dev_mem_t DummyComputeDevice::alloc_device_buffer(size_t size, int flags)
{
    return { malloc(size) };
}

void DummyComputeDevice::free_host_buffer(host_mem_t m)
{
    free(m.ptr);
}

void DummyComputeDevice::free_device_buffer(dev_mem_t m)
{
    free(m.ptr);
}

void DummyComputeDevice::memwrite(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    memcpy(((uint8_t *) dev_buf.ptr) + offset, host_buf.ptr, size);
}

void DummyComputeDevice::memread(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    memcpy(host_buf.ptr, ((uint8_t *) dev_buf.ptr) + offset, size);
}

// vim: ts=8 sts=4 sw=4 et
