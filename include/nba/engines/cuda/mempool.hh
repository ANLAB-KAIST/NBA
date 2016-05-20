#ifndef __NBA_CUDA_MEMPOOL_HH__
#define __NBA_CUDA_MEMPOOL_HH__

#include <nba/engines/cuda/utils.hh>
#include <nba/core/mempool.hh>
#include <nba/core/offloadtypes.hh>
#include <cstdint>
#include <cassert>
#include <cuda.h>

namespace nba {

class CUDAMemoryPool : public MemoryPool<dev_mem_t>
{
public:
    explicit CUDAMemoryPool()
        : MemoryPool(), base(nullptr)
    { }

    explicit CUDAMemoryPool(size_t max_size)
        : MemoryPool(max_size), base(nullptr)
    { }

    explicit CUDAMemoryPool(size_t max_size, size_t align)
        : MemoryPool(max_size, align), base(nullptr)
    { }

    virtual ~CUDAMemoryPool()
    {
        destroy();
    }

    bool init()
    {
        cutilSafeCall(cudaMalloc((void **) &base, max_size));
        return true;
    }

    dev_mem_t get_base_ptr() const
    {
        return { (void *) ((uintptr_t) base + shifts) };
    }

    int alloc(size_t size, dev_mem_t &ptr)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            ptr.ptr = (void *) ((uintptr_t) base + shifts + offset);
        return ret;
    }

    void destroy()
    {
        if (base != NULL)
            cudaFree(base);
    }

private:
    void *base;
};

class CPUMemoryPool : public MemoryPool<host_mem_t>
{
public:
    explicit CPUMemoryPool(int cuda_flags)
        : MemoryPool(), base(nullptr), flags(cuda_flags), use_external(false)
    { }

    explicit CPUMemoryPool(size_t max_size, int cuda_flags)
        : MemoryPool(max_size), base(nullptr), flags(cuda_flags), use_external(false)
    { }

    explicit CPUMemoryPool(size_t max_size, size_t align, int cuda_flags)
        : MemoryPool(max_size, align), base(nullptr), flags(cuda_flags), use_external(false)
    { }

    virtual ~CPUMemoryPool()
    {
        destroy();
    }

    bool init()
    {
        cutilSafeCall(cudaHostAlloc((void **) &base, max_size,
                      this->flags));
        return true;
    }

    bool init_with_flags(void *ext_ptr, int flags)
    {
        if (ext_ptr != nullptr) {
            base = ext_ptr;
            use_external = true;
        } else {
            cutilSafeCall(cudaHostAlloc((void **) &base, max_size,
                          flags));
        }
        return true;
    }

    host_mem_t get_base_ptr() const
    {
        return { (void *) ((uintptr_t) base + shifts) };
    }

    int alloc(size_t size, host_mem_t &m)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            m.ptr = (void *) ((uintptr_t) base + shifts + offset);
        return ret;
    }

    void destroy()
    {
        if (base != NULL && !use_external)
            cudaFreeHost(base);
    }

private:
    void *base;
    int flags;
    bool use_external;
};

}
#endif

// vim: ts=8 sts=4 sw=4 et
