#ifndef __NBA_DUMMY_MEMPOOL_HH__
#define __NBA_DUMMY_MEMPOOL_HH__

#include <nba/core/mempool.hh>
#include <cstdint>
#include <cassert>

namespace nba {

class DummyCPUMemoryPool : public MemoryPool
{
public:
    DummyCPUMemoryPool() : MemoryPool(), base(NULL)
    {
    }

    virtual ~DummyCPUMemoryPool()
    {
        destroy();
    }

    virtual bool init(unsigned long size)
    {
        max_size = size;
        base = malloc(size);
        return true;
    }

    void *alloc(size_t size)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            return (void *) ((uint8_t *) base + offset);
        return NULL;
    }

    void destroy()
    {
        if (base != NULL) {
            free(base);
            base = NULL;
        }
    }

    void *get_base_ptr() const
    {
        return base;
    }

protected:
    void *base;
};

}
#endif

// vim: ts=8 sts=4 sw=4 et
