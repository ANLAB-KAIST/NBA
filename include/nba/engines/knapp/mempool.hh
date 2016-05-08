#ifndef __NBA_KNAPP_MEMPOOL_HH__
#define __NBA_KNAPP_MEMPOOL_HH__

#include <nba/core/mempool.hh>
#include <nba/core/offloadtypes.hh>
#include <nba/engines/knapp/rma.hh>
#include <cstdint>
#include <cassert>


namespace nba {

class RMAPeerMemoryPool : public MemoryPool<dev_mem_t>
{
public:
    RMAPeerMemoryPool(knapp::RMABuffer *rma_buffer)
        : MemoryPool(), _rma_buffer(rma_buffer), base(nullptr)
    {
        /* Use peer-side virtual address. */
        base = (void *) _rma_buffer->peer_va();
    }

    virtual ~RMAPeerMemoryPool()
    {
        destroy();
    }

    bool init()
    {
        return true;
    }

    dev_mem_t get_base_ptr() const
    {
        return { base };
    }

    int alloc(size_t size, dev_mem_t &ptr)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            ptr.ptr = (void *) ((uintptr_t) base + offset);
        return ret;
    }

    void destroy()
    {
        // do nothing.
    }

    knapp::RMABuffer *rma_buffer()
    {
        return _rma_buffer;
    }

private:
    knapp::RMABuffer *_rma_buffer;
    void *base;
};

class RMALocalMemoryPool : public MemoryPool<host_mem_t>
{
public:
    RMALocalMemoryPool(knapp::RMABuffer *rma_buffer)
        : MemoryPool(), _rma_buffer(rma_buffer), base(nullptr)
    {
        /* Use my local virtual address. */
        base = (void *) _rma_buffer->va();
    }

    virtual ~RMALocalMemoryPool()
    {
        destroy();
    }

    bool init()
    {
        return true;
    }

    host_mem_t get_base_ptr() const
    {
        return { base };
    }

    int alloc(size_t size, host_mem_t &ptr)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0)
            ptr.ptr = (void *) ((uintptr_t) base + offset);
        return ret;
    }

    void destroy()
    {
        // do nothing.
    }

    knapp::RMABuffer *rma_buffer()
    {
        return _rma_buffer;
    }

private:
    knapp::RMABuffer *_rma_buffer;
    void *base;
};

}
#endif // __NBA_KNAPP_MEMPOOL_HH__

// vim: ts=8 sts=4 sw=4 et
