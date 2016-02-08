#include <nba/core/intrinsic.hh>
#include <nba/engines/dummy/computecontext.hh>

using namespace std;
using namespace nba;

DummyComputeContext::DummyComputeContext(unsigned ctx_id, ComputeDevice *mother_device)
 : ComputeContext(ctx_id, mother_device)
{
    type_name = "dummy";
    size_t io_base_size = 5 * 1024 * 1024; // TODO: read from config
    NEW(node_id, io_base_ring, FixedRing<unsigned>,
        NBA_MAX_IO_BASES, node_id);
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        NEW(node_id, _dev_mempool_in[i], DummyCPUMemoryPool, io_base_size, CACHE_LINE_SIZE);
        NEW(node_id, _dev_mempool_out[i], DummyCPUMemoryPool, io_base_size, CACHE_LINE_SIZE);
        NEW(node_id, _cpu_mempool_in[i], DummyCPUMemoryPool, io_base_size, CACHE_LINE_SIZE);
        NEW(node_id, _cpu_mempool_out[i], DummyCPUMemoryPool, io_base_size, CACHE_LINE_SIZE);
        _dev_mempool_in[i]->init();
        _dev_mempool_out[i]->init();
        _cpu_mempool_in[i]->init();
        _cpu_mempool_out[i]->init();
    }
}

DummyComputeContext::~DummyComputeContext()
{
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        _dev_mempool_in[i]->destroy();
        _dev_mempool_out[i]->destroy();
        _cpu_mempool_in[i]->destroy();
        _cpu_mempool_out[i]->destroy();
    }
}

io_base_t DummyComputeContext::alloc_io_base()
{
    if (io_base_ring->empty()) return INVALID_IO_BASE;
    unsigned i = io_base_ring->front();
    io_base_ring->pop_front();
    return (io_base_t) i;
}

int DummyComputeContext::alloc_input_buffer(io_base_t io_base, size_t size,
                                            host_mem_t &host_ptr, dev_mem_t &dev_ptr)
{
    unsigned i = io_base;
    assert(0 == _cpu_mempool_in[i]->alloc(size, host_ptr.ptr));
    assert(0 == _dev_mempool_in[i]->alloc(size, dev_ptr.ptr));
    return 0;
}

int DummyComputeContext::alloc_output_buffer(io_base_t io_base, size_t size,
                                             host_mem_t &host_ptr, dev_mem_t &dev_ptr)
{
    unsigned i = io_base;
    assert(0 == _cpu_mempool_out[i]->alloc(size, host_ptr.ptr));
    assert(0 == _dev_mempool_out[i]->alloc(size, dev_ptr.ptr));
    return 0;
}

void DummyComputeContext::map_input_buffer(io_base_t io_base, size_t offset, size_t len,
                                           host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.ptr = (void *) ((uintptr_t) _cpu_mempool_in[i]->get_base_ptr() + offset);
    dbuf.ptr = (void *) ((uintptr_t) _cpu_mempool_in[i]->get_base_ptr() + offset);
    // len is ignored.
}

void DummyComputeContext::map_output_buffer(io_base_t io_base, size_t offset, size_t len,
                                            host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    hbuf.ptr = (void *) ((uintptr_t)_cpu_mempool_out[i]->get_base_ptr() + offset);
    dbuf.ptr = (void *) ((uintptr_t)_cpu_mempool_out[i]->get_base_ptr() + offset);
    // len is ignored.
}

void *DummyComputeContext::unwrap_host_buffer(const host_mem_t hbuf) const
{
    return hbuf.ptr;
}

void *DummyComputeContext::unwrap_device_buffer(const dev_mem_t dbuf) const
{
    return dbuf.ptr;
}

size_t DummyComputeContext::get_input_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_in[i]->get_alloc_size();
}

size_t DummyComputeContext::get_output_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_out[i]->get_alloc_size();
}

void DummyComputeContext::clear_io_buffers(io_base_t io_base)
{
    unsigned i = io_base;
    _cpu_mempool_in[i]->reset();
    _cpu_mempool_out[i]->reset();
    _dev_mempool_in[i]->reset();
    _dev_mempool_out[i]->reset();
    io_base_ring->push_back(i);
}

int DummyComputeContext::enqueue_memwrite_op(const host_mem_t host_buf,
                                             const dev_mem_t dev_buf,
                                             size_t offset, size_t size)
{
    return 0;
}

int DummyComputeContext::enqueue_memread_op(const host_mem_t host_buf,
                                            const dev_mem_t dev_buf,
                                            size_t offset, size_t size)
{
    return 0;
}

int DummyComputeContext::enqueue_kernel_launch(dev_kernel_t kernel,
                                               struct resource_param *res)
{
    return 0;
}

int DummyComputeContext::enqueue_event_callback(
        void (*func_ptr)(ComputeContext *ctx, void *user_arg),
        void *user_arg)
{
    func_ptr(this, user_arg);
    return 0;
}


// vim: ts=8 sts=4 sw=4 et
