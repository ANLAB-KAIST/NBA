#ifndef __NSHADER_DUMMY_MEMPOOL_HH__
#define __NSHADER_DUMMY_MEMPOOL_HH__

#include <stdint.h>
#include <assert.h>
#include "../../lib/mempool.hh"

using namespace std;
using namespace nshader;

namespace nshader {

class DummyCPUMemoryPool : public MemoryPool
{
public:
    DummyCPUMemoryPool() : MemoryPool(), base_(NULL)
    {
    }

    virtual ~DummyCPUMemoryPool()
    {
        destroy();
    }

    virtual bool init(unsigned long size)
    {
        max_size_ = size;
        base_ = malloc(size);
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
        if (base_ != NULL) {
            free(base_);
            base_ = NULL;
        }
    }

    void *get_base_ptr()
    {
        return base_;
    }

protected:
    void *base_;
};

}
#endif

// vim: ts=8 sts=4 sw=4 et
