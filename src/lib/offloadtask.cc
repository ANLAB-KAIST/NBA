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
}

OffloadTask::~OffloadTask()
{
}

#if DEBUG
#define _debug_print_inb(tag, batch, dbid) { \
    void *begin_h; \
    memory_t begin_d; \
    cctx->get_input_current_pos(io_base, &begin_h, &begin_d); \
    size_t len = cctx->get_input_size(io_base) - last_input_size; \
    last_input_size = cctx->get_input_size(io_base); \
    printf("task[%lu, %p:%u] alloc_input_buffer (" tag ") %p:%d -> start:%p, end:%p, len:%'lu(0x%lx)\n", \
           task_id, cctx, (unsigned) io_base, batch, dbid, \
           (void *)((uintptr_t)begin_h - len), begin_h, len, len); \
}
#define _debug_print_outb(tag, batch, dbid) { \
    void *begin_h; \
    memory_t begin_d; \
    cctx->get_output_current_pos(io_base, &begin_h, &begin_d); \
    size_t len = cctx->get_output_size(io_base) - last_output_size; \
    last_output_size = cctx->get_output_size(io_base); \
    printf("task[%lu, %p:%u] alloc_output_buffer (" tag ") %p:%d -> start:%p, end:%p, len:%'lu(0x%lx)\n", \
           task_id, cctx, (unsigned) io_base, batch, dbid, \
           (void *)((uintptr_t)begin_h - len), begin_h, len, len); \
}
#else
#define _debug_print_inb(tag, batch, dbid)
#define _debug_print_outb(tag, batch, dbid)
#endif

void OffloadTask::prepare_read_buffer()
{
    // write: host-to-device input
    // read: device-to-host output
    cctx->get_input_current_pos(io_base, &host_write_begin, &dev_write_begin);
    cctx->get_output_current_pos(io_base, &host_read_begin, &dev_read_begin);
    input_alloc_size_begin  = cctx->get_input_size(io_base);
    output_alloc_size_begin = cctx->get_output_size(io_base);
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
                                             (void **) &t->aligned_item_sizes_h,
                                             &t->aligned_item_sizes_d);
                }
                _debug_print_inb("prepare_read_buffer.WHOLE", nullptr, dbid);
            } else if (rri.type == READ_PARTIAL_PACKET) {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    cctx->alloc_input_buffer(io_base, sizeof(uint64_t),
                                             (void **) &t->aligned_item_sizes_h,
                                             &t->aligned_item_sizes_d);
                }
                _debug_print_inb("prepare_read_buffer.PARTIAL", nullptr, dbid);
            } else {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    t->aligned_item_sizes_h = nullptr;
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
                    t->host_in_ptr    = nullptr;
                    t->dev_in_ptr.ptr = nullptr;
                }
            } else {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    tie(t->in_size, t->in_count) = db->calc_read_buffer_size(batch);
                    if (t->in_size > 0 && t->in_count > 0) {
                        cctx->alloc_input_buffer(io_base, t->in_size,
                                                 (void **) &t->host_in_ptr, &t->dev_in_ptr);
                        db->preprocess(batch, t->host_in_ptr);
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
                    t->host_out_ptr    = nullptr;
                    t->dev_out_ptr.ptr = nullptr;
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
                        t->host_out_ptr    = nullptr;
                        t->dev_out_ptr.ptr = nullptr;
                        tie(t->out_size, t->out_count) = db->calc_write_buffer_size(batch);
                        if (t->out_size > 0 && t->out_count > 0) {
                            cctx->alloc_output_buffer(io_base, t->out_size,
                                                      (void **) &t->host_out_ptr,
                                                      &t->dev_out_ptr);
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
    size_t dbarray_size = sizeof(struct datablock_kernel_arg *) * datablocks.size();
    cctx->alloc_input_buffer(io_base, dbarray_size, (void **) &dbarray_h, &dbarray_d);
    _debug_print_inb("copy_h2d.dbarray", nullptr, 0);
    assert(dbarray_h != nullptr);

    for (int dbid : datablocks) {
        int dbid_d = dbid_h2d[dbid];
        assert(dbid_d < (signed) datablocks.size());
        DataBlock *db = comp_ctx->datablock_registry[dbid];
        struct read_roi_info rri;
        struct write_roi_info wri;
        db->get_read_roi(&rri);
        db->get_write_roi(&wri);

        struct datablock_kernel_arg *dbarg_h;
        memory_t dbarg_d;
        size_t dbarg_size = sizeof(struct datablock_kernel_arg)
                            + batches.size() * sizeof(struct datablock_batch_info);
        cctx->alloc_input_buffer(io_base, dbarg_size, (void **) &dbarg_h, &dbarg_d);
        assert(dbarg_h != nullptr);

        dbarray_h[dbid_d] = (struct datablock_kernel_arg *) dbarg_d.ptr;
        dbarg_h->total_item_count_in  = 0;
        dbarg_h->total_item_count_out = 0;

        for (auto&& p : enumerate(batches)) {
            size_t b = p.first;
            PacketBatch *&batch = p.second;
            assert(batch->datablock_states != nullptr);
            struct datablock_tracker *t = &batch->datablock_states[dbid];

            if (rri.type == READ_WHOLE_PACKET && t->in_count > 0) {
                /* We need to copy the size array because each item may
                 * have different lengths. */
                assert(t->aligned_item_sizes_h != nullptr);
                dbarg_h->batches[b].item_sizes_in  = (uint16_t *)
                        ((char *) t->aligned_item_sizes_d.ptr
                         + (uintptr_t) offsetof(struct item_size_info, sizes));
                dbarg_h->batches[b].item_sizes_out = (uint16_t *)
                        ((char *) t->aligned_item_sizes_d.ptr
                         + (uintptr_t) offsetof(struct item_size_info, sizes));
                dbarg_h->batches[b].item_offsets_in = (dev_offset_t *)
                        ((char *) t->aligned_item_sizes_d.ptr
                         + (uintptr_t) offsetof(struct item_size_info, offsets));
                dbarg_h->batches[b].item_offsets_out = (dev_offset_t *)
                        ((char *) t->aligned_item_sizes_d.ptr
                         + (uintptr_t) offsetof(struct item_size_info, offsets));
            } else {
                /* Same for all batches.
                 * We assume the module developer knows the fixed length
                 * when writing device kernel codes. */
                dbarg_h->item_size_in  = rri.length;
                dbarg_h->item_size_out = wri.length;
                dbarg_h->batches[b].item_offsets_in  = nullptr;
                dbarg_h->batches[b].item_offsets_out = nullptr;
            }
            dbarg_h->batches[b].buffer_bases_in = t->dev_in_ptr.ptr;   // FIXME: generalize to CL?
            dbarg_h->batches[b].item_count_in   = t->in_count;
            dbarg_h->total_item_count_in       += t->in_count;
            dbarg_h->batches[b].buffer_bases_out = t->dev_out_ptr.ptr; // FIXME: generalize to CL?
            dbarg_h->batches[b].item_count_out   = t->out_count;
            dbarg_h->total_item_count_out       += t->out_count;
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

    uint16_t *batch_ids_h = nullptr;
    memory_t batch_ids_d;
    uint16_t *item_ids_h = nullptr;
    memory_t item_ids_d;

    for (PacketBatch *batch : batches) {
        struct datablock_tracker *t = &batch->datablock_states[dbid];
        all_item_count += t->in_count;
    }

    if (all_item_count > 0) {

        cctx->alloc_input_buffer(io_base, sizeof(uint16_t) * all_item_count,
                                 (void **) &batch_ids_h, &batch_ids_d);
        _debug_print_inb("execute.batch_ids", nullptr, 0);
        assert(batch_ids_h != nullptr);
        cctx->alloc_input_buffer(io_base, sizeof(uint16_t) * all_item_count,
                                 (void **) &item_ids_h, &item_ids_d);
        _debug_print_inb("execute.item_ids", nullptr, 0);
        assert(item_ids_h != nullptr);
        res.num_workitems = all_item_count;
        res.num_threads_per_workgroup = 256;
        res.num_workgroups = (all_item_count + res.num_threads_per_workgroup - 1)
                             / res.num_threads_per_workgroup;
        uint16_t batch_id = 0;
        unsigned global_idx = 0;
        for (PacketBatch *batch : batches) {
            struct datablock_tracker *t = &batch->datablock_states[dbid];
            for (unsigned item_id = 0; item_id < t->in_count; item_id ++) {
                batch_ids_h[global_idx] = batch_id;
                item_ids_h[global_idx]  = item_id;
                global_idx ++;
            }
            batch_id ++;
        }

        size_t last_alloc_size = cctx->get_input_size(io_base);
        //printf("GPU-offload-h2d-size: %'lu bytes\n", last_alloc_size);
        // ipv4@64B: 16K ~ 24K
        // ipsec@64B: ~ 5M
        cctx->enqueue_memwrite_op(host_write_begin, dev_write_begin, 0,
                                  last_alloc_size - input_alloc_size_begin);
        //cctx->enqueue_memwrite_op(host_write_begin, dev_write_begin, 0,
        //                          2097152);

        cctx->clear_checkbits();
        cctx->clear_kernel_args();

        state = TASK_EXECUTING;

        /* Framework-provided kernel arguments:
         * (1) array of datablock_kernel_arg[] indexed by datablock ID
         * (2) the number of batches
         */
        void *checkbits_d = cctx->get_device_checkbits();
        struct kernel_arg arg;
        arg = {(void *) &dbarray_d.ptr, sizeof(void *), alignof(void *)};
        cctx->push_kernel_arg(arg);
        arg = {(void *) &all_item_count, sizeof(uint32_t), alignof(uint32_t)};
        cctx->push_kernel_arg(arg);
        arg = {(void *) &batch_ids_d.ptr, sizeof(void *), alignof(void *)};
        cctx->push_kernel_arg(arg);
        arg = {(void *) &item_ids_d.ptr, sizeof(void *), alignof(void *)};
        cctx->push_kernel_arg(arg);
        arg = {(void *) &checkbits_d, sizeof(void *), alignof(void *)};
        cctx->push_kernel_arg(arg);

        offload_compute_handler &handler = elem->offload_compute_handlers[cctx->type_name];
        handler(cctx, &res);
        /* Skip kernel execution. */
        //res.num_workitems = 0;
        //res.num_threads_per_workgroup = 1;
        //res.num_workgroups = 1;
        //cctx->get_host_checkbits()[0] = 1;

    } else {

        /* Skip kernel execution. */
        res.num_workitems = 0;
        res.num_threads_per_workgroup = 1;
        res.num_workgroups = 1;
        cctx->get_host_checkbits()[0] = 1;
    }
}

bool OffloadTask::copy_d2h()
{
    state = TASK_D2H_COPYING;

    /* Coalesced D2H data copy. */
    size_t last_alloc_size = cctx->get_output_size(io_base);
    cctx->enqueue_memread_op(host_read_begin, dev_read_begin,
                             0, last_alloc_size - output_alloc_size_begin);
    return true;
}

bool OffloadTask::poll_kernel_finished()
{
    uint8_t *checkbits = cctx->get_host_checkbits();
    if (checkbits == nullptr) {
        return true;
    }
    for (unsigned i = 0; i < res.num_workgroups; i++) {
        if (checkbits[i] == 0) {
            return false;
        }
    }
    return true;
}

bool OffloadTask::poll_d2h_copy_finished()
{
    bool result = cctx->query();
    return result;
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
            int b = 0;
            for (PacketBatch *batch : batches) {
                struct datablock_tracker *t = &batch->datablock_states[dbid];
                if (t->host_out_ptr != nullptr) {
                    // FIXME: let the element to choose the datablock used for postprocessing,
                    //        or pass multiple datablocks that have outputs.
                    db->postprocess(elem, input_ports[b], batch, t->host_out_ptr);
                }
                b++;
            }
        } /* endif(check_postproc) */
    } /* endfor(dbid) */
}

// vim: ts=8 sts=4 sw=4 et
