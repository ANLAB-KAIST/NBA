#include "computecontext.hh"
#include "../../lib/log.hh"
#include "../../lib/common.hh"

using namespace std;
using namespace nba;

DummyComputeContext::DummyComputeContext(unsigned ctx_id, ComputeDevice *mother_device)
 : ComputeContext(ctx_id, mother_device)
{
    type_name = "dummy";
    size_t mem_size = 8 * 1024 * 1024; // TODO: read from config
    _dev_mempool_in.init(mem_size);
    _dev_mempool_out.init(mem_size);
    _cpu_mempool_in.init(mem_size);
    _cpu_mempool_out.init(mem_size);
}

DummyComputeContext::~DummyComputeContext()
{
    _dev_mempool_in.destroy();
    _dev_mempool_in.destroy();
    _cpu_mempool_out.destroy();
    _cpu_mempool_out.destroy();
}

int DummyComputeContext::alloc_input_buffer(size_t size, void **host_ptr, memory_t *dev_mem)
{
    *host_ptr = _cpu_mempool_in.alloc(size);
    dev_mem->ptr = _dev_mempool_in.alloc(size);
    return 0;
}

int DummyComputeContext::alloc_output_buffer(size_t size, void **host_ptr, memory_t *dev_mem)
{
    *host_ptr = _cpu_mempool_out.alloc(size);
    dev_mem->ptr = _dev_mempool_out.alloc(size);
    return 0;
}

void DummyComputeContext::clear_io_buffers()
{
    _cpu_mempool_in.reset();
    _cpu_mempool_out.reset();
    _dev_mempool_in.reset();
    _dev_mempool_out.reset();
}

void *DummyComputeContext::get_host_input_buffer_base()
{
    return _cpu_mempool_in.get_base_ptr();
}

memory_t DummyComputeContext::get_device_input_buffer_base()
{
    memory_t ret;
    ret.ptr = _dev_mempool_in.get_base_ptr();
    return ret;
}

size_t DummyComputeContext::get_total_input_buffer_size()
{
    return _cpu_mempool_in.get_alloc_size();
}

void DummyComputeContext::set_io_buffers(void *in_h, memory_t in_d, size_t in_sz,
                                         void *out_h, memory_t out_d, size_t out_sz)
{
    this->in_h = in_h;
    this->in_d = in_d;
    this->out_h = out_h;
    this->out_d = out_d;
    this->in_sz = in_sz;
    this->out_sz = out_sz;
}

void DummyComputeContext::set_io_buffer_elemsizes(size_t *in_h, memory_t in_d, size_t in_sz,
                                                  size_t *out_h, memory_t out_d, size_t out_sz)
{
    this->in_elemsizes_h   = in_h;
    this->in_elemsizes_d   = in_d;
    this->out_elemsizes_h  = out_h;
    this->out_elemsizes_d  = out_d;
    this->in_elemsizes_sz  = in_sz;
    this->out_elemsizes_sz = out_sz;
}

int DummyComputeContext::enqueue_memwrite_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    return 0;
}

int DummyComputeContext::enqueue_memread_op(void *host_buf, memory_t dev_buf, size_t offset, size_t size)
{
    return 0;
}

int DummyComputeContext::enqueue_kernel_launch(kernel_t kernel, struct resource_param *res,
                          struct kernel_arg *args, size_t num_args)
{
    return 0;
}

int DummyComputeContext::enqueue_event_callback(void (*func_ptr)(ComputeContext *ctx, void *user_arg), void *user_arg)
{
    func_ptr(this, user_arg);
    return 0;
}


// vim: ts=8 sts=4 sw=4 et
