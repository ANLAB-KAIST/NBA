#ifndef __NBA_PHI_MEMPOOL_HH__
#define __NBA_PHI_MEMPOOL_HH__

#include <CL/opencl.h>
#include <nba/core/mempool.hh>
#include <nba/engines/phi/utils.hh>
#include <cstdint>
#include <cstdlib>
#include <cassert>

namespace nba {

class PhiMemoryPool : public MemoryPool
{
public:
    PhiMemoryPool(cl_context clctx, cl_command_queue clqueue, int direction_hint)
     : MemoryPool(), clctx(clctx), clqueue(clqueue), direction_hint(direction_hint)
    {
    }

    virtual ~PhiMemoryPool()
    {
        destroy();
    }

    virtual bool init(unsigned long max_size)
    {
        max_size = max_size;
        cl_int err_ret;
        clbuf = clCreateBuffer(clctx, CL_MEM_HOST_NO_ACCESS |
                          (direction_hint == HOST_TO_DEVICE ? CL_MEM_READ_ONLY : CL_MEM_WRITE_ONLY),
                       max_size, NULL, &err_ret);
        if (err_ret != CL_SUCCESS)
            return false;
        return true;
    }

    memory_t alloc(size_t size)
    {
        memory_t ret;
        size_t offset;
        int r =_alloc(size, &offset);
        if (r == 0) {
            cl_buffer_region region = { offset, size };
            cl_int err_ret;
            // Create a sub-buffer inheriting all flags from clbuf.
            ret.clmem = clCreateSubBuffer(clbuf, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err_ret);
            if (err_ret != CL_SUCCESS) {
                fprintf(stderr, "clCreateSubBuffer() failed!\n");
                ret.ptr = NULL;
            }
            return ret;
        }
        ret.ptr = NULL;
        return ret;
    }

    virtual void destroy()
    {
        clReleaseMemObject(clbuf);
    }

    void *get_base_ptr() const
    {
        // TODO: convert clbuf to void*
        assert(false, "not implemented yet");
        return nullptr;
    }

private:
    cl_context clctx;
    cl_command_queue clqueue;
    int direction_hint;
    cl_mem clbuf;
};

class CPUMemoryPool : public MemoryPool
{
public:
    virtual ~CPUMemoryPool()
    {
        destroy();
    }

    virtual bool init(size_t max_size)
    {
        void *ret = NULL;
        max_size = max_size;
        base = (uint8_t*) malloc(max_size);
        return ret;
    }

    void *alloc(size_t size)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            return (void *) ((uint8_t *) base + offset);
        return NULL;
    }

    virtual void destroy()
    {
        if (base)
            free(base);
    }

    void *get_base_ptr() const
    {
        return base;
    }

private:
    void *base;
};

}
#endif

// vim: ts=8 sts=4 sw=4 et
