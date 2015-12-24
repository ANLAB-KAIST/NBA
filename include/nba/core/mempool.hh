#ifndef __NBA_MEMPOOL_HH__
#define __NBA_MEMPOOL_HH__

#include <nba/core/intrinsic.hh>
#include <cstdint>
#include <cassert>
#include <cerrno>

namespace nba
{

/**
 * This abstract memory pool class provides a bump allocator
 * in the memroy region defined by its subclasses.
 */
class MemoryPool
{
public:
    MemoryPool() : max_size(0), cur_pos(0)
    {}

    virtual ~MemoryPool() {}

    virtual bool init(size_t max_size) = 0;

    int _alloc(size_t size, size_t *start_offset)
    {
        if (cur_pos + size > max_size)
            return -ENOMEM;
        /* IMPORTANT: We need to return the position before adding the new size. */
        if (start_offset != nullptr)
            *start_offset = cur_pos;
        cur_pos += size;
        cur_pos = ALIGN_CEIL(cur_pos, CACHE_LINE_SIZE);
        return 0;
    }

    // The device implementer's should provide his own alloc() method.

    void reset()
    {
        cur_pos = 0;
    }

    size_t get_alloc_size() const
    {
        return cur_pos;
    }

    virtual void *get_base_ptr() const = 0;

protected:
    size_t max_size;
    size_t cur_pos;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
