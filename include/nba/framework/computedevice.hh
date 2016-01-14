#ifndef __NBA_COMPUTEDEVICE_HH__
#define __NBA_COMPUTEDEVICE_HH__

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <nba/core/offloadtypes.hh>
#include <nba/core/threading.hh>
#include <ev.h>

namespace nba {

enum HostMemoryFlags {
    HOST_DEFAULT = 0,
    HOST_PINNED = 1,
    HOST_MAPPED = 2,
    HOST_WRITECOMBINED = 4,
};

enum DeviceMemoryFlags {
    DEVICE_GLOBAL = 0,
};

struct compute_device_spec {
    unsigned node_id;
    int max_threads;
    int max_workgroups;
    int max_concurrent_kernels;
    size_t global_memory_size;
    // TODO: add more if needed
};

struct compute_device_util {
    /** A relative number in [0..1] indicating utilization of computation capacity. */
    float utilization;

    /** The number of used bytes in its global memory. */
    uint64_t used_memory_bytes;
};

class ComputeContext; /* forward declaration */

class ComputeDevice {

    /**
     * A ComputeDevice represents a physical unit of processor.
     * For example, a socket of CPU and a card of GPU are separate
     * ComputeDevice instances.
     */

public:
    friend class ComputeContext;

    ComputeDevice(unsigned node_id, unsigned device_id, size_t num_contexts)
                  : type_name("<invalid>"),
                    input_watcher(NULL), available_sema(num_contexts),
                    node_id(node_id), device_id(device_id), num_contexts(num_contexts),
                    _lock()
    {
    }
    virtual ~ComputeDevice() {}

    virtual int get_spec(struct compute_device_spec *spec) = 0;
    virtual int get_utilization(struct compute_device_util *util) = 0;

    ComputeContext *get_available_context()
    {
        _lock.acquire();
        uint64_t ret = available_sema.try_down();
        if (ret == 0) {
            _lock.release();
            return NULL;
        }
        ComputeContext *cctx = this->_get_available_context();
        _lock.release();
        return cctx;
    }

    void return_context(ComputeContext *cctx)
    {
        _lock.acquire();
        this->_return_context(cctx);
        /* Notify the coprocessor thread that we have avilable contexts now. */
        available_sema.up();
        _lock.release();
    }

    std::vector<ComputeContext *> &get_contexts()
    {
        return contexts;
    }

    virtual void *alloc_host_buffer(size_t size, int flags) = 0;
    virtual memory_t alloc_device_buffer(size_t size, int flags) = 0;
    virtual void free_host_buffer(void *ptr) = 0;
    virtual void free_device_buffer(memory_t ptr) = 0;

    /* Synchronous versions */
    virtual void memwrite(void *host_buf, memory_t dev_buf, size_t offset, size_t size) = 0;
    virtual void memread(void *host_buf, memory_t dev_buf, size_t offset, size_t size) = 0;

    std::string type_name;
    struct ev_async *input_watcher;
    AsyncSemaphore available_sema;

    const unsigned node_id;
    const unsigned device_id;
    const size_t num_contexts;
protected:
    std::vector<ComputeContext *> contexts;
    Lock _lock;

    virtual ComputeContext *_get_available_context() = 0;
    virtual void _return_context(ComputeContext *cctx) = 0;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
