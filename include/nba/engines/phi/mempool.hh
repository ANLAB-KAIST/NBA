#ifndef __NBA_CL_MEMPOOL_HH__
#define __NBA_CL_MEMPOOL_HH__

#include <CL/opencl.h>
#include <nba/core/mempool.hh>
#include <nba/engines/phi/utils.hh>
#include <cstdint>
#include <cstdlib>
#include <cassert>

namespace nba {

class CLMemoryPool : public MemoryPool<cl_mem>
{
public:
    CLMemoryPool(size_t max_size, size_t align, cl_context clctx, cl_command_queue clqueue, int direction_hint)
        : MemoryPool(max_size, align), clctx(clctx), clqueue(clqueue), direction_hint(direction_hint)
    { }

    virtual ~CLMemoryPool()
    {
        destroy();
    }

    bool init()
    {
        cl_int err_ret;
        // Let OpenCL runtime to allocate both host-side buffer
        // and device-side buffer using its own optimized flags.
        base_buf = clCreateBuffer(clctx, CL_MEM_ALLOC_HOST_PTR |
                          (direction_hint == HOST_TO_DEVICE
                           ? (CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY)
                           : (CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY)),
                       max_size, nullptr, &err_ret);
        if (err_ret != CL_SUCCESS)
            return false;
        return true;
    }

    cl_mem get_base_ptr() const
    {
        return base_buf;
    }

    int alloc(size_t size, cl_mem &subbuf)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0) {
            cl_buffer_region region = { offset, size };
            cl_int err;
            // Create a sub-buffer inheriting all flags from base_buf.
            subbuf = clCreateSubBuffer(base_buf, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
            if (err != CL_SUCCESS) {
                ret = -ENOMEM;
            }
        }
        return ret;
    }

    void destroy()
    {
        // TODO: on reset(), release sub-buffer objects as well.
        clReleaseMemObject(base_buf);
    }

private:
    cl_context clctx;
    cl_command_queue clqueue;
    int direction_hint;
    cl_mem base_buf;
};

}
#endif

// vim: ts=8 sts=4 sw=4 et
