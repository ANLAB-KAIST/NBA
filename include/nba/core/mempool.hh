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
template<typename T>
class MemoryPool
{
public:
    MemoryPool()
        : max_size(0), align(CACHE_LINE_SIZE), cur_pos(0)
    { }

    MemoryPool(size_t max_size)
        : max_size(max_size), align(CACHE_LINE_SIZE), cur_pos(0)
    { }

    MemoryPool(size_t max_size, size_t align)
        : max_size(max_size), align(align), cur_pos(0)
    { }

    virtual ~MemoryPool() { }

    size_t get_alloc_size() const { return cur_pos; }

    virtual bool init() = 0;

    virtual T get_base_ptr() const = 0;

    // Device implementers should provide his own alloc() method,
    // using the inherited _alloc() method which provides new offset
    // calculation according to bump allocator strategy.
    virtual int alloc(size_t size, T& ptr) = 0;

    virtual void destroy() = 0;

    // We implement a bump allocator.
    void reset() { cur_pos = 0; }

protected:
    int _alloc(size_t size, size_t *start_offset)
    {
        if (ALIGN_CEIL(cur_pos + size, align) > max_size)
            return -ENOMEM;
        /* IMPORTANT: We need to return the position before adding the new size. */
        if (start_offset != nullptr)
            *start_offset = cur_pos;
        cur_pos += size;
        cur_pos = ALIGN_CEIL(cur_pos, align);
        return 0;
    }

    const size_t max_size;
    const size_t align;
    size_t cur_pos;
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
