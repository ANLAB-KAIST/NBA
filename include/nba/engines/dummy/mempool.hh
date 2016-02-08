#ifndef __NBA_DUMMY_MEMPOOL_HH__
#define __NBA_DUMMY_MEMPOOL_HH__

#include <nba/core/mempool.hh>
#include <cstdint>
#include <cassert>

namespace nba {

class DummyCPUMemoryPool : public MemoryPool<void *>
{
public:
    DummyCPUMemoryPool()
        : MemoryPool(), base(nullptr)
    { }

    DummyCPUMemoryPool(size_t max_size)
        : MemoryPool(max_size), base(nullptr)
    { }

    DummyCPUMemoryPool(size_t max_size, size_t align)
        : MemoryPool(max_size, align), base(nullptr)
    { }

    virtual ~DummyCPUMemoryPool()
    {
        destroy();
    }

    bool init()
    {
        base = malloc(max_size);
        return true;
    }

    int alloc(size_t size, void *&ptr)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            ptr = (void *) ((uint8_t *) base + offset);
        return ret;
    }

    void destroy()
    {
        if (base != nullptr) {
            free(base);
            base = nullptr;
        }
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
