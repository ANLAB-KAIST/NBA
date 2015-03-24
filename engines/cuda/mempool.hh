#ifndef __NSHADER_CUDA_MEMPOOL_HH__
#define __NSHADER_CUDA_MEMPOOL_HH__

#include <stdint.h>
#include <assert.h>
#include <cuda.h>
#include "utils.hh"
#include "../../lib/mempool.hh"

using namespace std;
using namespace nshader;

namespace nshader {

class CUDAMemoryPool : public MemoryPool
{
public:
    CUDAMemoryPool() : MemoryPool(), base_(NULL)
    {
    }

    virtual ~CUDAMemoryPool()
    {
        destroy();
    }

    virtual bool init(size_t max_size)
    {
        max_size_ = max_size;
        cutilSafeCall(cudaMalloc((void **) &base_, max_size));
        return true;
    }

    void *alloc(size_t size)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            return (void *) ((uint8_t *) base_ + offset);
        return NULL;
    }

    void destroy()
    {
        if (base_ != NULL)
            cudaFree(base_);
    }

    void *get_base_ptr()
    {
        return base_;
    }

private:
    void *base_;
};

class CPUMemoryPool : public MemoryPool
{
public:
    CPUMemoryPool(int cuda_flags) : MemoryPool(), base_(NULL), flags_(cuda_flags)
    {
    }

    virtual ~CPUMemoryPool()
    {
        destroy();
    }

    virtual bool init(unsigned long size)
    {
        max_size_ = size;
        cutilSafeCall(cudaHostAlloc((void **) &base_, size,
                      flags_));
        return true;
    }

    void *alloc(size_t size)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            return (void *) ((uint8_t *) base_ + offset);
        return NULL;
    }

    void destroy()
    {
        if (base_ != NULL)
            cudaFreeHost(base_);
    }

    void *get_base_ptr()
    {
        return base_;
    }

protected:
    void *base_;
    int flags_;
};

}
#endif

// vim: ts=8 sts=4 sw=4 et
