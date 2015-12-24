#include <nba/core/intrinsic.hh>
#include <nba/engines/dummy/computecontext.hh>

using namespace std;
using namespace nba;

DummyComputeContext::DummyComputeContext(unsigned ctx_id, ComputeDevice *mother_device)
 : ComputeContext(ctx_id, mother_device)
{
    type_name = "dummy";
    size_t io_base_size = 5 * 1024 * 1024; // TODO: read from config
    io_base_ring.init(NBA_MAX_IO_BASES, node_id, io_base_ring_buf);
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        _dev_mempool_in[i].init(io_base_size);
        _dev_mempool_out[i].init(io_base_size);
        _cpu_mempool_in[i].init(io_base_size);
        _cpu_mempool_out[i].init(io_base_size);
    }
}

DummyComputeContext::~DummyComputeContext()
{
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        _dev_mempool_in[i].destroy();
        _dev_mempool_out[i].destroy();
        _cpu_mempool_in[i].destroy();
        _cpu_mempool_out[i].destroy();
    }
}

io_base_t DummyComputeContext::alloc_io_base()
{
    if (io_base_ring.empty()) return INVALID_IO_BASE;
    unsigned i = io_base_ring.front();
    io_base_ring.pop_front();
    return (io_base_t) i;
}


void DummyComputeContext::get_input_current_pos(io_base_t io_base, void **host_ptr, memory_t *dev_mem) const
{
    unsigned i = io_base;
    *host_ptr = (char*)_cpu_mempool_in[i].get_base_ptr() + (uintptr_t)_cpu_mempool_in[i].get_alloc_size();
    dev_mem->ptr = (char*)_dev_mempool_in[i].get_base_ptr() + (uintptr_t)_dev_mempool_in[i].get_alloc_size();
}

void DummyComputeContext::get_output_current_pos(io_base_t io_base, void **host_ptr, memory_t *dev_mem) const
{
    unsigned i = io_base;
    *host_ptr = (char*)_cpu_mempool_out[i].get_base_ptr() + (uintptr_t)_cpu_mempool_out[i].get_alloc_size();
    dev_mem->ptr = (char*)_dev_mempool_out[i].get_base_ptr() + (uintptr_t)_dev_mempool_out[i].get_alloc_size();
}

size_t DummyComputeContext::get_input_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_in[i].get_alloc_size();
}

size_t DummyComputeContext::get_output_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _cpu_mempool_in[i].get_alloc_size();
}

int DummyComputeContext::alloc_input_buffer(io_base_t io_base, size_t size, void **host_ptr, memory_t *dev_mem)
{
    unsigned i = io_base;
    *host_ptr = _cpu_mempool_in[i].alloc(size);
    assert(*host_ptr != nullptr);
    dev_mem->ptr = _dev_mempool_in[i].alloc(size);
    assert(dev_mem->ptr != nullptr);
    return 0;
}

int DummyComputeContext::alloc_output_buffer(io_base_t io_base, size_t size, void **host_ptr, memory_t *dev_mem)
{
    unsigned i = io_base;
    *host_ptr = _cpu_mempool_out[i].alloc(size);
    assert(*host_ptr != nullptr);
    dev_mem->ptr = _dev_mempool_out[i].alloc(size);
    assert(dev_mem->ptr != nullptr);
    return 0;
}

void DummyComputeContext::clear_io_buffers(io_base_t io_base)
{
    unsigned i = io_base;
    _cpu_mempool_in[i].reset();
    _cpu_mempool_out[i].reset();
    _dev_mempool_in[i].reset();
    _dev_mempool_out[i].reset();
    io_base_ring.push_back(i);
}

int DummyComputeContext::enqueue_memwrite_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    return 0;
}

int DummyComputeContext::enqueue_memread_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    return 0;
}

int DummyComputeContext::enqueue_kernel_launch(kernel_t kernel, struct resource_param *res)
{
    return 0;
}

int DummyComputeContext::enqueue_event_callback(void (*func_ptr)(ComputeContext *ctx, void *user_arg), void *user_arg)
{
    func_ptr(this, user_arg);
    return 0;
}


// vim: ts=8 sts=4 sw=4 et
