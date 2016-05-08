#include <nba/core/intrinsic.hh>
#include <nba/core/enumerate.hh>
#include <nba/framework/logging.hh>
#include <nba/framework/datablock.hh>
#include <nba/framework/elementgraph.hh>
#include <nba/framework/offloadtask.hh>
#include <nba/framework/computedevice.hh>
#include <nba/framework/computecontext.hh>
#ifdef USE_CUDA
#include <nba/engines/cuda/computedevice.hh>
#include <nba/engines/cuda/computecontext.hh>
#endif
#include <nba/core/shiftedint.hh> // should come after cuda headers
#include <tuple>
#include <ev.h>
#include <rte_memcpy.h>
#include <rte_ether.h>
#include <rte_prefetch.h>
#include <netinet/ip.h>

using namespace std;
using namespace nba;

static thread_local char dummy_buffer[NBA_MAX_PACKET_SIZE] = {0,};

OffloadTask::OffloadTask()
{
    datablocks.clear();
    batches.clear();
    input_ports.clear();
    elemgraph = nullptr;
    src_loop = nullptr;
    comp_ctx = nullptr;
    coproc_ctx = nullptr;
    completion_watcher = nullptr;
    completion_queue   = nullptr;
    cctx = nullptr;
    io_base = INVALID_IO_BASE;
    offload_start = 0;
    num_pkts = 0;
    num_bytes = 0;
    // for debugging
    last_input_size = 0;
    last_output_size = 0;
    kernel_skipped = false;
}

OffloadTask::~OffloadTask()
{
}

#ifdef DEBUG_OFFLOAD
#define _debug_print_inb(tag, batch, dbid) { \
    size_t end = cctx->get_input_size(io_base); \
    size_t len = end - last_input_size; \
    size_t begin = end - len; \
    last_input_size = end; \
    printf("task[%lu, %p:%u] alloc_input_buffer (" tag ") %p:%d " \
           "-> start:0x%08x, end:0x%08x, len:%lu (0x%lx) bytes\n", \
           task_id, cctx, (unsigned) io_base, batch, dbid, \
           begin, end, len, len); \
}
#define _debug_print_outb(tag, batch, dbid) { \
    size_t end = cctx->get_output_size(io_base); \
    size_t len = end - last_output_size; \
    size_t begin = end - len; \
    last_output_size = end; \
    printf("task[%lu, %p:%u] alloc_output_buffer (" tag ") %p:%d " \
           "-> start:0x%08x, end:0x%08x, len:%lu (0x%lx) bytes\n", \
           task_id, cctx, (unsigned) io_base, batch, dbid, \
           begin, end, len, len); \
}
#else
#define _debug_print_inb(tag, batch, dbid)
#define _debug_print_outb(tag, batch, dbid)
#endif

void OffloadTask::prepare_read_buffer()
{
    // write: host-to-device input
    // read: device-to-host output
    input_begin  = cctx->get_input_size(io_base);
    output_begin = cctx->get_output_size(io_base);
    _debug_print_inb("at-beginning", nullptr, 0);
    _debug_print_outb("at-beginning", nullptr, 0);

    for (int dbid : datablocks) {
        if (elemgraph->check_preproc(elem, dbid)) {
            DataBlock *db = comp_ctx->datablock_registry[dbid];
            struct read_roi_info rri;
            db->get_read_roi(&rri);
            if (rri.type == READ_WHOLE_PACKET) {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    cctx->alloc_input_buffer(io_base, sizeof(struct item_size_info),
                                             t->aligned_item_sizes_h,
                                             t->aligned_item_sizes_d);
                    t->aligned_item_sizes = (struct item_size_info *)
                            cctx->unwrap_host_buffer(t->aligned_item_sizes_h);
                }
                _debug_print_inb("prepare_read_buffer.WHOLE", nullptr, dbid);
            } else if (rri.type == READ_PARTIAL_PACKET) {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    cctx->alloc_input_buffer(io_base, sizeof(uint64_t),
                                             t->aligned_item_sizes_h,
                                             t->aligned_item_sizes_d);
                    t->aligned_item_sizes = (struct item_size_info *)
                            cctx->unwrap_host_buffer(t->aligned_item_sizes_h);
                }
                _debug_print_inb("prepare_read_buffer.PARTIAL", nullptr, dbid);
            } else {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    //t->aligned_item_sizes_h = nullptr;
                }
            }
        } /* endif(check_preproc) */
    } /* endfor(dbid) */
    for (int dbid : datablocks) {
        if (elemgraph->check_preproc(elem, dbid)) {
            DataBlock *db = comp_ctx->datablock_registry[dbid];
            struct read_roi_info rri;
            db->get_read_roi(&rri);
            if (rri.type == READ_NONE) {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    t->in_size = 0;
                    t->in_count = 0;
                }
            } else {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    tie(t->in_size, t->in_count) = db->calc_read_buffer_size(batch);
                    // Now aligned_item_sizes has valid values.
                    if (t->in_size > 0 && t->in_count > 0) {
                        cctx->alloc_input_buffer(io_base, t->in_size,
                                                 t->host_in_ptr, t->dev_in_ptr);
                        void *inp = cctx->unwrap_host_buffer(t->host_in_ptr);
                        db->preprocess(batch, inp);
                    }
                }
                _debug_print_inb("prepare_read_buffer.preproc", nullptr, dbid);
            } /* endcase(rri.type) */
        } /* endif(check_preproc) */
    } /* endfor(dbid) */
}

void OffloadTask::prepare_write_buffer()
{
    for (int dbid : datablocks) {
        if (elemgraph->check_preproc(elem, dbid)) {
            DataBlock *db = comp_ctx->datablock_registry[dbid];
            struct write_roi_info wri;
            struct read_roi_info rri;
            db->get_write_roi(&wri);
            db->get_read_roi(&rri);
            if (wri.type == WRITE_NONE) {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    t->out_size = 0;
                    t->out_count = 0;
                }
            } else {
                //if (rri.type == READ_WHOLE_PACKET && wri.type == WRITE_WHOLE_PACKET) {
                //    for (PacketBatch *batch : batches) {
                //        struct datablock_tracker *t = &batch->datablock_states[dbid];
                //        /* Reuse read_roi currently. Do NOT update size & count here! */
                //        t->out_size  = t->in_size;
                //        t->out_count = t->in_count;
                //        t->host_out_ptr = t->host_in_ptr;
                //        t->dev_out_ptr  = t->dev_in_ptr;
                //    }
                //} else {
                    for (PacketBatch *batch : batches) {
                        struct datablock_tracker *t = &batch->datablock_states[dbid];
                        tie(t->out_size, t->out_count) = db->calc_write_buffer_size(batch);
                        if (t->out_size > 0 && t->out_count > 0) {
                            cctx->alloc_output_buffer(io_base, t->out_size,
                                                      t->host_out_ptr,
                                                      t->dev_out_ptr);
                        }
                    }
                    _debug_print_outb("prepare_write_buffer", nullptr, dbid);
                //} /* endif(rri.type, wri.type) */
            } /* endif(wri.type) */
        } /* endif(check_preproc) */
    } /* endfor(dbid) */
}

bool OffloadTask::copy_h2d()
{
    state = TASK_H2D_COPYING;

    /* Copy the datablock information for the first kernel argument. */
    size_t dbarray_size = sizeof(void *) * datablocks.size();
    struct datablock_kernel_arg **dbarray;
    cctx->alloc_input_buffer(io_base, dbarray_size, dbarray_h, dbarray_d);
    _debug_print_inb("copy_h2d.dbarray", nullptr, 0);
    dbarray = (struct datablock_kernel_arg **) cctx->unwrap_host_buffer(dbarray_h);

    for (int dbid : datablocks) {
        int dbid_d = dbid_h2d[dbid];
        assert(dbid_d < (signed) datablocks.size());
        DataBlock *db = comp_ctx->datablock_registry[dbid];
        struct read_roi_info rri;
        struct write_roi_info wri;
        db->get_read_roi(&rri);
        db->get_write_roi(&wri);

        struct datablock_kernel_arg *dbarg;
        host_mem_t dbarg_h;
        dev_mem_t dbarg_d;
        size_t dbarg_size = sizeof(struct datablock_kernel_arg)
                            + batches.size() * sizeof(struct datablock_batch_info);
        cctx->alloc_input_buffer(io_base, dbarg_size, dbarg_h, dbarg_d);
        dbarg = (struct datablock_kernel_arg *) cctx->unwrap_host_buffer(dbarg_h);
        dbarray[dbid_d] = (struct datablock_kernel_arg *) cctx->unwrap_device_buffer(dbarg_d);
        dbarg->total_item_count  = 0;

        // NOTE: To use our "datablock kernel arg" data structures,
        //       the underlying kernel language must support generic
        //       pointer references.
        //       (e.g,. NVIDIA CUDA / OpenCL 2.0+)

        for (auto&& p : enumerate(batches)) {
            size_t b = p.first;
            PacketBatch *&batch = p.second;
            assert(batch->datablock_states != nullptr);
            struct datablock_tracker *t = &batch->datablock_states[dbid];

            if (rri.type == READ_WHOLE_PACKET && t->in_count > 0) {
                /* We need to copy the size array because each item may
                 * have different lengths. */
                //assert(t->aligned_item_sizes_h != nullptr);
                uintptr_t base_ptr = (uintptr_t) cctx->unwrap_device_buffer(t->aligned_item_sizes_d);
                dbarg->batches[b].item_sizes  = (uint16_t *)
                        (base_ptr + offsetof(struct item_size_info, sizes));
                dbarg->batches[b].item_offsets = (dev_offset_t *)
                        (base_ptr + offsetof(struct item_size_info, offsets));
            } else {
                /* Same for all batches.
                 * We assume the module developer knows the fixed length
                 * when writing device kernel codes. */
                if (rri.type != READ_NONE)
                    dbarg->item_size  = rri.length;
                if (wri.type != WRITE_NONE)
                    dbarg->item_size = wri.length;
                dbarg->batches[b].item_offsets  = nullptr;
            }
            if (rri.type != READ_NONE) {
                dbarg->batches[b].buffer_bases = cctx->unwrap_device_buffer(t->dev_in_ptr);
                dbarg->batches[b].item_count   = t->in_count;
                dbarg->total_item_count       += t->in_count;
            }
            if (wri.type != WRITE_NONE) {
                dbarg->batches[b].buffer_bases = cctx->unwrap_device_buffer(t->dev_out_ptr);
                dbarg->batches[b].item_count   = t->out_count;
                dbarg->total_item_count       += t->out_count;
            }
        } /* endfor(batches) */
    } /* endfor(dbid) */
    return true;
}

/**
 * Execute the offload handler and copy back the output data
 * (device-to-host).
 */

void OffloadTask::execute()
{
    /* The element sets which dbid becomes the reference point that
     * provides the number of total "items".
     * In most cases the unit will be packets, but sometimes (like
     * IPsecAES) it will be a custom units such as 16 B blocks. */

    uint32_t all_item_count = 0;
    uint32_t num_batches = batches.size();
    int dbid = elem->get_offload_item_counter_dbid();
    DataBlock *db = comp_ctx->datablock_registry[dbid];

    host_mem_t item_counts_h;
    dev_mem_t item_counts_d;

    for (PacketBatch *batch : batches) {
        struct datablock_tracker *t = &batch->datablock_states[dbid];
        all_item_count += t->in_count;
    }

    if (all_item_count > 0) {

        cctx->alloc_input_buffer(io_base, sizeof(uint32_t) * batches.size(),
                                 item_counts_h, item_counts_d);
        _debug_print_inb("execute.item_counts", nullptr, 0);
        uint32_t *item_counts = (uint32_t *) cctx->unwrap_host_buffer(item_counts_h);
        uint32_t num_batches = batches.size();
        res.num_workitems = all_item_count;
        res.num_threads_per_workgroup = elem->get_desired_workgroup_size(cctx->type_name.c_str());
        res.num_workgroups = (all_item_count + res.num_threads_per_workgroup - 1)
                             / res.num_threads_per_workgroup;
        for (auto&& pair : enumerate(batches)) {
            item_counts[pair.first] = (pair.second)->datablock_states[dbid].in_count;
        }

        size_t total_input_size = cctx->get_input_size(io_base) - input_begin;
        //printf("GPU-offload-h2d-size: %'lu bytes\n", total_input_size);
        // ipv4@64B: 16K ~ 24K
        // ipsec@64B: ~ 5M
        host_mem_t host_input;
        dev_mem_t dev_input;
        cctx->map_input_buffer(io_base, input_begin, total_input_size,
                               host_input, dev_input);
        cctx->enqueue_memwrite_op(host_input, dev_input, 0, total_input_size);

        cctx->clear_kernel_args();

        state = TASK_EXECUTING;

        /* Add framework-provided kernel arguments:
         * (1) array of datablock_kernel_arg[] indexed by datablock ID
         * (2) the number of batches
         */
        void *ptr_args[3]; // storage for rvalue
        struct kernel_arg arg;

        ptr_args[0] = cctx->unwrap_device_buffer(dbarray_d);
        arg = {&ptr_args[0], sizeof(void *), alignof(void *)};
        cctx->push_kernel_arg(arg);

        arg = {(void *) &all_item_count, sizeof(uint32_t), alignof(uint32_t)};
        cctx->push_kernel_arg(arg);

        ptr_args[1] = cctx->unwrap_device_buffer(item_counts_d);
        arg = {&ptr_args[1], sizeof(void *), alignof(void *)};
        cctx->push_kernel_arg(arg);

        arg = {(void *) &num_batches, sizeof(uint32_t), alignof(uint32_t)};
        cctx->push_kernel_arg(arg);

        /* Add ComputeContext-provided kernel arguments. */
        cctx->push_common_kernel_args();

        /* Add element-provided kernel arguments and let the element
         * initate launch. */
        kernel_skipped = false;
        offload_compute_handler &handler = elem->offload_compute_handlers[cctx->type_name];
        handler(cctx->mother(), cctx, &res);

    } else {

        /* Skip kernel execution. */
        kernel_skipped = true;
    }
}

bool OffloadTask::copy_d2h()
{
    state = TASK_D2H_COPYING;

    /* Coalesced D2H data copy. */
    size_t total_output_size = cctx->get_output_size(io_base) - output_begin;
    host_mem_t host_output;
    dev_mem_t dev_output;
    cctx->map_output_buffer(io_base, output_begin, total_output_size,
                            host_output, dev_output);
    cctx->enqueue_memread_op(host_output, dev_output, 0, total_output_size);
    return true;
}

bool OffloadTask::poll_kernel_finished()
{
    return kernel_skipped || cctx->poll_kernel_finished();
}

bool OffloadTask::poll_d2h_copy_finished()
{
    return cctx->poll_output_finished();
}

void OffloadTask::notify_completion()
{
    /* Notify the computation thread. */
    state = TASK_FINISHED;
    assert(0 == rte_ring_sp_enqueue(completion_queue, (void *) this));
    ev_async_send(src_loop, completion_watcher);
}

void OffloadTask::postprocess()
{
    for (int dbid : datablocks) {
        if (elemgraph->check_postproc(elem, dbid)) {
            DataBlock *db = comp_ctx->datablock_registry[dbid];
            struct write_roi_info wri;
            db->get_write_roi(&wri);
            for (auto&& pair : enumerate(batches)) {
                PacketBatch *batch = pair.second;
                struct datablock_tracker *t = &batch->datablock_states[dbid];
                if (t->out_size > 0) {
                    // FIXME: let the element to choose the datablock used for postprocessing,
                    //        or pass multiple datablocks that have outputs.
                    void *outp = cctx->unwrap_host_buffer(t->host_out_ptr);
                    db->postprocess(elem, input_ports[pair.first], batch, outp);
                }
            }
        } /* endif(check_postproc) */
    } /* endfor(dbid) */
}

// vim: ts=8 sts=4 sw=4 et
