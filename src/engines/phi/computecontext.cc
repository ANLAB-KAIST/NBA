#include <nba/core/intrinsic.hh>
#include <nba/engines/phi/computecontext.hh>
#include <nba/engines/phi/computedevice.hh>
#include <unistd.h>

using namespace std;
using namespace nba;

struct phi_event_context {
    ComputeContext *computectx;
    void (*callback)(ComputeContext *ctx, void *user_arg);
    void *user_arg;
};

#define IO_BASE_SIZE (16 * 1024 * 1024)
#define IO_MEMPOOL_ALIGN (8)

PhiComputeContext::PhiComputeContext(unsigned ctx_id, ComputeDevice *mother_device)
 : ComputeContext(ctx_id, mother_device)
{
    type_name = "phi";
    size_t io_base_size = ALIGN_CEIL(IO_BASE_SIZE, getpagesize());
    cl_int err_ret;
    PhiComputeDevice *modevice = (PhiComputeDevice *) mother_device;
    clqueue = clCreateCommandQueue(modevice->clctx, modevice->cldevid, 0, &err_ret);
    if (err_ret != CL_SUCCESS) {
        rte_panic("clCreateCommandQueue()@PhiComputeContext() failed\n");
    }
    NEW(node_id, io_base_ring, FixedRing<unsigned>,
        NBA_MAX_IO_BASES, node_id);
    for (unsigned i = 0; i < NBA_MAX_IO_BASES; i++) {
        io_base_ring->push_back(i);
        NEW(node_id, _mempool_in[i], CLMemoryPool,
            io_base_size, IO_MEMPOOL_ALIGN, modevice->clctx, clqueue, HOST_TO_DEVICE),
        _mempool_in[i]->init();
        NEW(node_id, _mempool_out[i], CLMemoryPool,
            io_base_size, IO_MEMPOOL_ALIGN, modevice->clctx, clqueue, DEVICE_TO_HOST),
        _mempool_out[i]->init();
    }
    checkbits_d.clmem = clCreateBuffer(modevice->clctx, CL_MEM_READ_WRITE,
                                       MAX_BLOCKS, NULL, &err_ret);
    if (err_ret != CL_SUCCESS) {
        rte_panic("clCreateBuffer()@PhiComputeContext() failed to create the checkbit region!\n");
    }
    checkbits_h = clEnqueueMapBuffer(clqueue, checkbits_d.clmem, CL_TRUE,
                                     CL_MAP_READ | CL_MAP_WRITE,
                                     0, MAX_BLOCKS, NULL, NULL, &err_ret);
    if (err_ret != CL_SUCCESS) {
        rte_panic("clEnqueueMapBuffer()@PhiComputeContext() failed to map the checkbit region!\n");
    }
    clev = nullptr;
}

PhiComputeContext::~PhiComputeContext()
{
    for (unsigned i =0; i < NBA_MAX_IO_BASES; i++) {
        _mempool_in[i]->destroy();
        _mempool_out[i]->destroy();
    }
    clReleaseCommandQueue(clqueue);
}

io_base_t PhiComputeContext::alloc_io_base()
{
    if (io_base_ring->empty()) return INVALID_IO_BASE;
    unsigned i = io_base_ring->front();
    io_base_ring->pop_front();
    return (io_base_t) i;
}

int PhiComputeContext::alloc_input_buffer(io_base_t io_base, size_t size,
                                          host_mem_t &host_mem, dev_mem_t &dev_mem)
{
    unsigned i = io_base;
    assert(0 == _mempool_in[i]->alloc(size, dev_mem));
    assert(CL_SUCCESS == clGetMemObjectInfo(dev_mem.clmem, CL_MEM_HOST_PTR,
                                            sizeof(void *), &host_mem.ptr, nullptr));
    return 0;
}

int PhiComputeContext::alloc_output_buffer(io_base_t io_base, size_t size,
                                           host_mem_t &host_mem, dev_mem_t &dev_mem)
{
    unsigned i = io_base;
    assert(0 == _mempool_out[i]->alloc(size, dev_mem));
    assert(CL_SUCCESS == clGetMemObjectInfo(dev_mem.clmem, CL_MEM_HOST_PTR,
                                            sizeof(void *), &host_mem.ptr, nullptr));
    return 0;
}

void PhiComputeContext::map_input_buffer(io_base_t io_base, size_t offset, size_t len,
                                         host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    cl_mem base = _mempool_in[i]->get_base_ptr();
    cl_int err;
    cl_buffer_region region = { offset, len };
    dbuf.clmem = clCreateSubBuffer(base, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    assert(CL_SUCCESS == err);
    assert(CL_SUCCESS == clGetMemObjectInfo(dbuf.clmem, CL_MEM_HOST_PTR,
                                            sizeof(void *), &hbuf.ptr, nullptr));
}

void PhiComputeContext::map_output_buffer(io_base_t io_base, size_t offset, size_t len,
                                          host_mem_t &hbuf, dev_mem_t &dbuf) const
{
    unsigned i = io_base;
    cl_mem base = _mempool_out[i]->get_base_ptr();
    cl_int err;
    cl_buffer_region region = { offset, len };
    dbuf.clmem = clCreateSubBuffer(base, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    assert(CL_SUCCESS == err);
    assert(CL_SUCCESS == clGetMemObjectInfo(dbuf.clmem, CL_MEM_HOST_PTR,
                                            sizeof(void *), &hbuf.ptr, nullptr));
}

void *PhiComputeContext::unwrap_host_buffer(host_mem_t hbuf) const
{
    return hbuf.ptr;
}

void *PhiComputeContext::unwrap_device_buffer(dev_mem_t dbuf) const
{
    void *ptr = nullptr;
    clGetMemObjectInfo(dbuf.clmem, CL_MEM_HOST_PTR, sizeof(void *), &ptr, nullptr);
    return ptr;
}

size_t PhiComputeContext::get_input_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _mempool_in[i]->get_alloc_size();
}

size_t PhiComputeContext::get_output_size(io_base_t io_base) const
{
    unsigned i = io_base;
    return _mempool_out[i]->get_alloc_size();
}

void PhiComputeContext::clear_io_buffers(io_base_t io_base)
{
    unsigned i = io_base;
    _mempool_in[i]->reset();
    _mempool_out[i]->reset();
    io_base_ring->push_back(i);
}

void PhiComputeContext::clear_kernel_args()
{
    num_kernel_args = 0;
}

void PhiComputeContext::push_kernel_arg(struct kernel_arg &arg)
{
    assert(num_kernel_args < PHI_MAX_KERNEL_ARGS);
    kernel_args[num_kernel_args ++] = arg;  /* Copied to the array. */
}

int PhiComputeContext::enqueue_memwrite_op(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    return (int) clEnqueueWriteBuffer(clqueue, dev_buf.clmem, CL_FALSE, offset, size, host_buf.ptr, 0, NULL, &clev);
}

int PhiComputeContext::enqueue_memread_op(host_mem_t host_buf, dev_mem_t dev_buf, size_t offset, size_t size)
{
    return (int) clEnqueueReadBuffer(clqueue, dev_buf.clmem, CL_FALSE, offset, size, host_buf.ptr, 0, NULL, &clev);
}

int PhiComputeContext::enqueue_kernel_launch(dev_kernel_t kernel, struct resource_param *res)
{
    for (unsigned i = 0; i < num_kernel_args; i++) {
        phiSafeCall(clSetKernelArg(kernel.clkernel, 6 + i, kernel_args[i].size, kernel_args[i].ptr));
    }
    clear_checkbits(res->num_workgroups);
    state = ComputeContext::RUNNING;
    phiSafeCall(clEnqueueNDRangeKernel(clqueue, kernel.clkernel, 1, NULL,
                                       &res->num_workitems, &res->num_threads_per_workgroup,
                                       0, NULL, &clev));
    return 0;
}

int PhiComputeContext::enqueue_event_callback(void (*func_ptr)(ComputeContext *ctx, void *user_arg), void *user_arg)
{
    auto cb = [](cl_event ev, cl_int status, void *user_data)
    {
        assert(status == CL_COMPLETE);
        struct phi_event_context *cectx = (struct phi_event_context *) user_data;
        cectx->callback(cectx->computectx, cectx->user_arg);
        delete cectx;
    };
    // TODO: how to avoid using new/delete?
    struct phi_event_context *cectx = new struct phi_event_context;
    cectx->computectx = this;
    cectx->callback = func_ptr;
    cectx->user_arg = user_arg;
    phiSafeCall(clEnqueueMarker(clqueue, &clev_marker));
    phiSafeCall(clSetEventCallback(clev_marker, CL_COMPLETE, cb, cectx));
    return 0;
}


// vim: ts=8 sts=4 sw=4 et
