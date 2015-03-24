#ifndef __NBA_MEMPOOL_HH__
#define __NBA_MEMPOOL_HH__

#include <stdint.h>
#include <assert.h>

#define __ALIGN(x,a) (((x)+(a)-1)&~((a)-1))

namespace nba
{

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
            return -1;
        curpos_ = __ALIGN(curpos_, 64);
        if (start_offset)
            *start_offset = curpos_;
        curpos_ += size;
        curpos_ = __ALIGN(curpos_, 64);
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

#undef __ALIGN

#endif

// vim: ts=8 sts=4 sw=4 et
