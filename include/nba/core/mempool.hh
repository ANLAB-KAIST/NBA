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
    MemoryPool() : max_size_(0), curpos_(0)
    {}

    virtual ~MemoryPool() {}

    virtual bool init(size_t max_size) = 0;

    int _alloc(size_t size, size_t *start_offset)
    {
        if (curpos_ + size > max_size_)
            return -ENOMEM;
        /* IMPORTANT: We need to return the position before adding the new size. */
        if (start_offset != nullptr)
            *start_offset = curpos_;
        curpos_ += size;
        curpos_ = ALIGN_CEIL(curpos_, CACHE_LINE_SIZE);
        return 0;
    }

    // The device implementer's should provide his own alloc() method.

    void reset()
    {
        curpos_ = 0;
    }

    size_t get_alloc_size()
    {
        return curpos_;
    }

    virtual void *get_base_ptr() = 0;

protected:
    size_t max_size_;
    size_t curpos_;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
