#include <nba/core/intrinsic.hh>
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
#include <tuple>
#include <ev.h>
#include <rte_memcpy.h>
#include <rte_ether.h>
#include <rte_prefetch.h>
#include <netinet/ip.h>

using namespace std;
using namespace nba;

#define COALESCED_COPY
#undef  DEBUG_HOSTSIDE

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
}

OffloadTask::~OffloadTask()
{
}

void OffloadTask::prepare_read_buffer()
{
    assert(io_base != INVALID_IO_BASE);
    #ifdef COALESCED_COPY
    cctx->get_input_current_pos(io_base, &host_write_begin, &dev_write_begin);
    cctx->get_output_current_pos(io_base, &host_read_begin, &dev_read_begin);
    input_alloc_size_begin  = cctx->get_input_size(io_base);
    output_alloc_size_begin = cctx->get_output_size(io_base);
    #endif

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
            } else if (rri.type == READ_PARTIAL_PACKET) {
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    cctx->alloc_input_buffer(io_base, sizeof(uint64_t),
                                             (void **) &t->aligned_item_sizes_h,
                                             &t->aligned_item_sizes_d);
                }
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
                    t->host_in_ptr    = nullptr;
                    t->dev_in_ptr.ptr = nullptr;
                    if (t->in_size > 0 && t->in_count > 0) {
                        cctx->alloc_input_buffer(io_base, t->in_size,
                                                 (void **) &t->host_in_ptr, &t->dev_in_ptr);
                        assert(t->host_in_ptr != nullptr);
                        db->preprocess(batch, t->host_in_ptr);
                    }
                }
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
                for (PacketBatch *batch : batches) {
                    struct datablock_tracker *t = &batch->datablock_states[dbid];
                    t->host_out_ptr    = nullptr;
                    t->dev_out_ptr.ptr = nullptr;
                    if (rri.type == READ_WHOLE_PACKET && wri.type == WRITE_WHOLE_PACKET) {
                        /* Reuse read_roi currently. Do NOT update size & count here! */
                        t->out_size  = t->in_size;
                        t->out_count = t->in_count;
                        t->host_out_ptr = t->host_in_ptr;
                        t->dev_out_ptr  = t->dev_in_ptr;
                    } else {
                        tie(t->out_size, t->out_count) = db->calc_write_buffer_size(batch);
                        if (t->out_size > 0 && t->out_count > 0) {
                            cctx->alloc_output_buffer(io_base, t->out_size,
                                                      (void **) &t->host_out_ptr,
                                                      &t->dev_out_ptr);
                            assert(t->host_out_ptr != nullptr);
                        }
                    }
                }
            }
        } /* endif(check_preproc) */
    } /* endfor(dbid) */
}

bool OffloadTask::copy_h2d()
{
    bool has_h2d_copies = false;
    state = TASK_H2D_COPYING;

    /* Copy the datablock information for the first kernel argument. */
    size_t dbarray_size = ALIGN_CEIL(sizeof(struct datablock_kernel_arg) * datablocks.size(), CACHE_LINE_SIZE);
    cctx->alloc_input_buffer(io_base, dbarray_size, (void **) &dbarray_h, &dbarray_d);
    assert(dbarray_h != nullptr);

    #ifndef COALESCED_COPY
    void *item_info_h = nullptr;
    memory_t item_info_d;
    size_t item_info_size = 0;
    #endif

    for (int dbid : datablocks) {
        int dbid_d = dbid_h2d[dbid];
        dbarray_h[dbid_d].total_item_count_in  = 0;
        dbarray_h[dbid_d].total_item_count_out = 0;
        assert(dbid_d < (signed) datablocks.size());

        DataBlock *db = comp_ctx->datablock_registry[dbid];
        struct read_roi_info rri;
        struct write_roi_info wri;
        db->get_read_roi(&rri);
        db->get_write_roi(&wri);

        int b = 0;
        for (PacketBatch *batch : batches) {
            struct datablock_tracker *t = &batch->datablock_states[dbid];

            if (rri.type == READ_WHOLE_PACKET && t->in_count > 0) {
                /* We need to copy the size array because each item may
                 * have different lengths. */
                assert(t->aligned_item_sizes_h != nullptr);
                #ifndef COALESCED_COPY
                if (item_info_h == nullptr) {
                    item_info_h = t->aligned_item_sizes_h;
                    item_info_d = t->aligned_item_sizes_d;
                }
                item_info_size += ALIGN(sizeof(struct item_size_info), CACHE_LINE_SIZE);
                #endif
                dbarray_h[dbid_d].item_sizes_in[b]  = (uint16_t *) ((char *) t->aligned_item_sizes_d.ptr
                                                                    + (uintptr_t) offsetof(struct item_size_info, sizes));
                dbarray_h[dbid_d].item_sizes_out[b] = (uint16_t *) ((char *) t->aligned_item_sizes_d.ptr
                                                                    + (uintptr_t) offsetof(struct item_size_info, sizes));
                dbarray_h[dbid_d].item_offsets_in[b]  = (uint16_t *) ((char *) t->aligned_item_sizes_d.ptr
                                                                      + (uintptr_t) offsetof(struct item_size_info, offsets));
                dbarray_h[dbid_d].item_offsets_out[b] = (uint16_t *) ((char *) t->aligned_item_sizes_d.ptr
                                                                      + (uintptr_t) offsetof(struct item_size_info, offsets));
            } else {
                /* Same for all batches.
                 * We assume the module developer knows the fixed length
                 * when writing device kernel codes. */
                dbarray_h[dbid_d].item_size_in  = rri.length;
                dbarray_h[dbid_d].item_size_out = wri.length;
                dbarray_h[dbid_d].item_offsets_in[b]  = nullptr;
                dbarray_h[dbid_d].item_offsets_out[b] = nullptr;
            }
            dbarray_h[dbid_d].buffer_bases_in[b]   = t->dev_in_ptr.ptr;   // FIXME: generalize to CL?
            dbarray_h[dbid_d].item_count_in[b]     = t->in_count;
            dbarray_h[dbid_d].total_item_count_in += t->in_count;
            dbarray_h[dbid_d].buffer_bases_out[b]   = t->dev_out_ptr.ptr; // FIXME: generalize to CL?
            dbarray_h[dbid_d].item_count_out[b]     = t->out_count;
            dbarray_h[dbid_d].total_item_count_out += t->out_count;
            b++;
        } /* endfor(batches) */
    } /* endfor(dbid) */

    #ifndef COALESCED_COPY
    cctx->enqueue_memwrite_op(item_info_h, item_info_d, 0, item_info_size);
    // FIXME: hacking by knowing internal behaviour of cuda_mempool...
    cctx->enqueue_memwrite_op(dbarray_h, dbarray_d, 0, dbarray_size);
    #endif

    has_h2d_copies = true;

    /* Coalesced H2D data copy.
     * We need to check and copy not-yet-tranferred-to-GPU buffers one by
     * one, but it causes high device API call overheads.
     * We aggregate continuous copies to reduce the number of API calls.
     * If there are no reused datablocks, all copies are shrinked into a
     * single API call. */
    #ifndef COALESCED_COPY
    void *first_host_in_ptr = nullptr;
    memory_t first_dev_in_ptr;
    size_t total_size = 0;
    for (int dbid : datablocks) {
        if (elemgraph->check_preproc(elem, dbid)) {
            for (PacketBatch *batch : batches) {
                struct datablock_tracker *t = &batch->datablock_states[dbid];
                if (t == nullptr || t->host_in_ptr == nullptr) {
                    if (first_host_in_ptr != nullptr) {
                        /* Discontinued copy. */
                        cctx->enqueue_memwrite_op(first_host_in_ptr, first_dev_in_ptr,
                                                  0, total_size);
                        /* Reset. */
                        first_host_in_ptr = nullptr;
                        total_size        = 0;
                    }
                    continue;
                }
                if (t->in_count == 0) assert(t->in_size == 0);
                if (t->in_count > 0) assert(t->in_size > 0);
                /* IMPORTANT: IO buffer allocations are aligned by cache line size!! */
                if (first_host_in_ptr == nullptr) {
                    first_host_in_ptr = t->host_in_ptr;
                    first_dev_in_ptr  = t->dev_in_ptr;
                }
                if ((char*) first_host_in_ptr + (uintptr_t) total_size
                        != (char*) t->host_in_ptr)
                {
                    cctx->enqueue_memwrite_op(first_host_in_ptr, first_dev_in_ptr,
                                              0, total_size);
                    first_host_in_ptr = t->host_in_ptr;
                    total_size  = ALIGN(t->in_size, CACHE_LINE_SIZE);
                } else {
                    total_size += ALIGN(t->in_size, CACHE_LINE_SIZE);
                }
            }
        } /* endif(check_preproc) */
    } /* endfor(dbid) */
    if (first_host_in_ptr != nullptr) {
        /* Finished copy. */
        cctx->enqueue_memwrite_op(first_host_in_ptr, first_dev_in_ptr, 0, total_size);
    }
    #endif
    return has_h2d_copies;
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
        assert(batch_ids_h != nullptr);
        cctx->alloc_input_buffer(io_base, sizeof(uint16_t) * all_item_count,
                                 (void **) &item_ids_h, &item_ids_d);
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
        #ifdef COALESCED_COPY
        size_t last_alloc_size = cctx->get_input_size(io_base);
        cctx->enqueue_memwrite_op(host_write_begin, dev_write_begin, 0, last_alloc_size - input_alloc_size_begin);
        #else
        cctx->enqueue_memwrite_op(batch_ids_h, batch_ids_d, 0, ALIGN(sizeof(uint16_t) * all_item_count, CACHE_LINE_SIZE) * 2);
        #endif

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
    #ifdef COALESCED_COPY
    size_t last_alloc_size = cctx->get_output_size(io_base);
    cctx->enqueue_memread_op(host_read_begin, dev_read_begin,
                             0, last_alloc_size - output_alloc_size_begin);
    #else
    void *first_host_out_ptr = nullptr;
    memory_t first_dev_out_ptr;
    size_t total_size = 0;
    for (int dbid : datablocks) {
        if (elemgraph->check_postproc(elem, dbid)) {
            DataBlock *db = comp_ctx->datablock_registry[dbid];
            for (PacketBatch *batch : batches) {
                struct datablock_tracker *t = &batch->datablock_states[dbid];
                if (t == nullptr || t->host_out_ptr == nullptr
                        || t->out_count == 0 || t->out_size == 0)
                {
                    if (first_host_out_ptr != nullptr) {
                        /* Discontinued copy. */
                        cctx->enqueue_memread_op(first_host_out_ptr, first_dev_out_ptr, 0, total_size);
                        /* Reset. */
                        first_host_out_ptr = nullptr;
                        total_size         = 0;
                    }
                    continue;
                }
                //if (t->out_count == 0) assert(t->out_size == 0);
                //if (t->out_count > 0) assert(t->out_size > 0);
                if (first_host_out_ptr == nullptr) {
                    first_host_out_ptr = t->host_out_ptr;
                    first_dev_out_ptr  = t->dev_out_ptr;
                }
                if ((char*) first_host_out_ptr + (uintptr_t) total_size
                        != (char*) t->host_out_ptr)
                {
                    cctx->enqueue_memread_op(first_host_out_ptr, first_dev_out_ptr, 0, total_size);
                    first_host_out_ptr = t->host_out_ptr;
                    total_size  = ALIGN(t->out_size, CACHE_LINE_SIZE);
                } else {
                    total_size += ALIGN(t->out_size, CACHE_LINE_SIZE);
                }
            }
        } /* endif(check_postproc) */
    } /* endfor(dbid) */
    if (first_host_out_ptr != nullptr) {
        /* Finished copy. */
        cctx->enqueue_memread_op(first_host_out_ptr, first_dev_out_ptr, 0, total_size);
    }
    #endif
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

    if (elemgraph->check_postproc_all(elem)) {
        /* Reset all datablock trackers. */
        for (PacketBatch *batch : batches) {
            if (batch->datablock_states != nullptr) {
                struct datablock_tracker *t = batch->datablock_states;
                t->host_in_ptr = nullptr;
                t->host_out_ptr = nullptr;
                rte_mempool_put(comp_ctx->dbstate_pool, (void *) t);
                batch->datablock_states = nullptr;
            }
        }
        /* Release per-task io_base. */
        cctx->clear_io_buffers(io_base);
        //printf("%s task finished\n", elem->class_name());
    }
}

// vim: ts=8 sts=4 sw=4 et
