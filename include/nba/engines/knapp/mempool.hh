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
    RMAPeerMemoryPool(uint32_t buffer_id, knapp::RMABuffer *rma_buffer, size_t max_size)
        : MemoryPool(max_size), _buffer_id(buffer_id), _rma_buffer(rma_buffer), base(nullptr)
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
        dev_mem_t ret;
        ret.m = { _buffer_id, base };
        return ret;
    }

    int alloc(size_t size, dev_mem_t &dbuf)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0) {
            dbuf.m = {
                _buffer_id,
                (void *) ((uintptr_t) base + offset)
            };
        }
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
    uint32_t _buffer_id;
    knapp::RMABuffer *_rma_buffer;
    void *base;
};

class RMALocalMemoryPool : public MemoryPool<host_mem_t>
{
public:
    RMALocalMemoryPool(uint32_t buffer_id, knapp::RMABuffer *rma_buffer, size_t max_size)
        : MemoryPool(max_size), _buffer_id(buffer_id), _rma_buffer(rma_buffer), base(nullptr)
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
        host_mem_t ret;
        ret.m = { _buffer_id, base };
        return ret;
    }

    int alloc(size_t size, host_mem_t &hbuf)
    {
        size_t offset;
        int ret = _alloc(size, &offset);
        if (ret == 0) {
            hbuf.m = {
                _buffer_id,
                (void *) ((uintptr_t) base + offset)
            };
        }
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
    uint32_t _buffer_id;
    knapp::RMABuffer *_rma_buffer;
    void *base;
};

}
#endif // __NBA_KNAPP_MEMPOOL_HH__

// vim: ts=8 sts=4 sw=4 et
