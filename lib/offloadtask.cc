#include "common.hh"
#include "log.hh"
#include "datablock.hh"
#include "offloadtask.hh"
#include "computedevice.hh"
#include "computecontext.hh"
#ifdef USE_CUDA
#include "../engines/cuda/computedevice.hh"
#include "../engines/cuda/computecontext.hh"
#endif
#include "elementgraph.hh"

#include <rte_common.h>
#include <rte_memcpy.h>
#include <rte_ether.h>
#include <rte_prefetch.h>
#include <tuple>
#include <netinet/ip.h>

using namespace std;
using namespace nba;

#define COALESC_COPY
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
    offload_start = 0;
    num_pkts = 0;
    num_bytes = 0;
}

OffloadTask::~OffloadTask()
{
}

void OffloadTask::prepare_read_buffer()
{
    for (int dbid : datablocks) {
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
                    cctx->alloc_input_buffer(t->in_size, (void **) &t->host_in_ptr, &t->dev_in_ptr);
                    assert(t->host_in_ptr != nullptr);
                    db->preprocess(batch, t->host_in_ptr);
                } else
                    printf("EMPTY BATCH @ prepare_read_buffer()\n");
            }
        }
    }
}

void OffloadTask::prepare_write_buffer()
{
    for (int dbid : datablocks) {
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
                        cctx->alloc_output_buffer(t->out_size, (void **) &t->host_out_ptr, &t->dev_out_ptr);
                        assert(t->host_out_ptr != nullptr);
                    } else
                        printf("EMPTY BATCH @ prepare_write_buffer()\n");
                }
            }
        }
    }
}

bool OffloadTask::copy_h2d()
{
    bool has_h2d_copies = false;

    /* Copy the datablock information for the first kernel argument. */
    size_t dbarray_size = ALIGN(sizeof(struct datablock_kernel_arg) * datablocks.size(), CACHE_LINE_SIZE);
    cctx->alloc_input_buffer(dbarray_size, (void **) &dbarray_h, &dbarray_d);
    assert(dbarray_h != nullptr);
    size_t itemszarray_size = 0;

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

            if (rri.type == READ_WHOLE_PACKET) {
                /* We need to copy the size array because each item may
                 * have different lengths. */
                uint16_t *item_sizes_h;
                memory_t item_sizes_d;
                cctx->alloc_input_buffer(t->in_count * sizeof(uint16_t), (void **) &item_sizes_h, &item_sizes_d);
                assert(item_sizes_h != nullptr);
                itemszarray_size += ALIGN(t->in_count * sizeof(uint16_t), CACHE_LINE_SIZE);
                rte_memcpy(item_sizes_h, &t->aligned_item_sizes.sizes[0], sizeof(uint16_t) * t->in_count);
                #ifdef DEBUG_HOSTSIDE
                dbarray_h[dbid_d].item_sizes_in[b]  = (uint16_t *) item_sizes_h;
                dbarray_h[dbid_d].item_sizes_out[b] = (uint16_t *) item_sizes_h;
                #else
                dbarray_h[dbid_d].item_sizes_in[b]  = (uint16_t *) item_sizes_d.ptr;
                dbarray_h[dbid_d].item_sizes_out[b] = (uint16_t *) item_sizes_d.ptr;
                #endif

                uint16_t *item_offsets_h;
                memory_t item_offsets_d;
                cctx->alloc_input_buffer(t->in_count * sizeof(uint16_t), (void **) &item_offsets_h, &item_offsets_d);
                assert(item_sizes_h != nullptr);
                itemszarray_size += ALIGN(t->in_count * sizeof(uint16_t), CACHE_LINE_SIZE);
                rte_memcpy(item_offsets_h, &t->aligned_item_sizes.offsets[0], sizeof(uint16_t) * t->in_count);
                #ifdef DEBUG_HOSTSIDE
                dbarray_h[dbid_d].item_offsets_in[b]  = (uint16_t *) item_offsets_h;
                dbarray_h[dbid_d].item_offsets_out[b] = (uint16_t *) item_offsets_h;
                #else
                dbarray_h[dbid_d].item_offsets_in[b]  = (uint16_t *) item_offsets_d.ptr;
                dbarray_h[dbid_d].item_offsets_out[b] = (uint16_t *) item_offsets_d.ptr;
                #endif

            } else {
                /* Same for all batches.
                 * We assume the module developer knows the fixed length
                 * when writing device kernel codes. */
                dbarray_h[dbid_d].item_size_in  = rri.length;
                dbarray_h[dbid_d].item_size_out = wri.length;
                dbarray_h[dbid_d].item_offsets_in[b]  = nullptr;
                dbarray_h[dbid_d].item_offsets_out[b] = nullptr;
            }
            #ifdef DEBUG_HOSTSIDE
            dbarray_h[dbid_d].buffer_bases_in[b]   = t->host_in_ptr;
            #else
            dbarray_h[dbid_d].buffer_bases_in[b]   = t->dev_in_ptr.ptr;   // FIXME: generalize to CL?
            #endif
            dbarray_h[dbid_d].item_count_in[b]     = t->in_count;
            dbarray_h[dbid_d].total_item_count_in += t->in_count;
            #ifdef DEBUG_HOSTSIDE
            dbarray_h[dbid_d].buffer_bases_out[b]   = t->host_out_ptr;
            #else
            dbarray_h[dbid_d].buffer_bases_out[b]   = t->dev_out_ptr.ptr; // FIXME: generalize to CL?
            #endif
            dbarray_h[dbid_d].item_count_out[b]     = t->out_count;
            dbarray_h[dbid_d].total_item_count_out += t->out_count;
            b++;
        }
    }
    // FIXME: hacking by knowing internal behaviour of cuda_mempool...
    cctx->enqueue_memwrite_op(dbarray_h, dbarray_d, 0, dbarray_size + itemszarray_size);
    has_h2d_copies = true;

    /* Coalesced H2D data copy. */
    void *first_host_in_ptr = nullptr;
    int copies = 0;
    memory_t first_dev_in_ptr;
    size_t total_size = 0;
    for (int dbid : datablocks) {
        int b = 0;
        for (PacketBatch *batch : batches) {
            struct datablock_tracker *t = &batch->datablock_states[dbid];
            if (t == nullptr || t->host_in_ptr == nullptr || t->in_count == 0 || t->in_size == 0) {
                #ifdef COALESC_COPY
                if (first_host_in_ptr != nullptr) {
                    /* Discontinued copy. */
                    cctx->enqueue_memwrite_op(first_host_in_ptr, first_dev_in_ptr, 0, total_size);
                    /* Reset. */
                    first_host_in_ptr = nullptr;
                    total_size        = 0;
                }
                #endif
                continue;
            }
            //if (t->in_count == 0) assert(t->in_size == 0);
            //if (t->in_count > 0) assert(t->in_size > 0);
            if (first_host_in_ptr == nullptr) {
                first_host_in_ptr = t->host_in_ptr;
                first_dev_in_ptr  = t->dev_in_ptr;
            }
            total_size += ALIGN(t->in_size, CACHE_LINE_SIZE);
            #ifndef COALESC_COPY
            cctx->enqueue_memwrite_op(t->host_in_ptr, t->dev_in_ptr, 0, t->in_size);
            #endif
            has_h2d_copies = true;
            b++;
        }
    }
    #ifdef COALESC_COPY
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

        cctx->alloc_input_buffer(sizeof(uint16_t) * all_item_count, (void **) &batch_ids_h, &batch_ids_d);
        assert(batch_ids_h != nullptr);
        cctx->alloc_input_buffer(sizeof(uint16_t) * all_item_count, (void **) &item_ids_h, &item_ids_d);
        assert(item_ids_h != nullptr);
        res.num_workitems = all_item_count;
        res.num_threads_per_workgroup = 256;
        res.num_workgroups = (all_item_count + res.num_threads_per_workgroup - 1) / res.num_threads_per_workgroup;
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
        cctx->enqueue_memwrite_op(batch_ids_h, batch_ids_d, 0, ALIGN(sizeof(uint16_t) * all_item_count, CACHE_LINE_SIZE) * 2);

        cctx->clear_checkbits();
        cctx->clear_kernel_args();

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
    /* Coalesced D2H data copy. */
    bool has_d2h_copies = false;
    void *first_host_out_ptr = nullptr;
    int copies = 0;
    memory_t first_dev_out_ptr;
    size_t total_size = 0;
    for (int dbid : datablocks) {
        DataBlock *db = comp_ctx->datablock_registry[dbid];
        for (PacketBatch *batch : batches) {
            struct datablock_tracker *t = &batch->datablock_states[dbid];
            if (t == nullptr || t->host_out_ptr == nullptr || t->out_count == 0 || t->out_size == 0) {
                #ifdef COALESC_COPY
                if (first_host_out_ptr != nullptr) {
                    /* Discontinued copy. */
                    cctx->enqueue_memread_op(first_host_out_ptr, first_dev_out_ptr, 0, total_size);
                    /* Reset. */
                    first_host_out_ptr = nullptr;
                    total_size         = 0;
                }
                #endif
                continue;
            }
            //if (t->out_count == 0) assert(t->out_size == 0);
            //if (t->out_count > 0) assert(t->out_size > 0);
            if (first_host_out_ptr == nullptr) {
                first_host_out_ptr = t->host_out_ptr;
                first_dev_out_ptr  = t->dev_out_ptr;
            }
            total_size += ALIGN(t->out_size, CACHE_LINE_SIZE);
            #ifndef COALESC_COPY
            cctx->enqueue_memread_op(t->host_out_ptr, t->dev_out_ptr, 0, t->out_size);
            #endif
            has_d2h_copies = true;
        }
    }
    #ifdef COALESC_COPY
    if (first_host_out_ptr != nullptr) {
        /* Finished copy. */
        cctx->enqueue_memread_op(first_host_out_ptr, first_dev_out_ptr, 0, total_size);
    }
    #endif
    return has_d2h_copies;
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
    assert(0 == rte_ring_sp_enqueue(completion_queue, (void *) this));
    ev_async_send(src_loop, completion_watcher);
}

void OffloadTask::postprocess()
{
    for (int dbid : datablocks) {
        DataBlock *db = comp_ctx->datablock_registry[dbid];
        struct write_roi_info wri;
        db->get_write_roi(&wri);
        if (wri.type == WRITE_NONE)
            continue;
        int b = 0;
        bool done = false;
        for (PacketBatch *batch : batches) {
            // TODO: query the elemgraph analysis result
            //       to check if elem is the postproc point
            //       of the current datablock.
            struct datablock_tracker *t = &batch->datablock_states[dbid];
            if (t->host_out_ptr != nullptr) {
                // FIXME: let the element to choose the datablock used for postprocessing,
                //        or pass multiple datablocks that have outputs.
                db->postprocess(elem, input_ports[b], batch, t->host_out_ptr);
                done = true;
            }
            b++;
        }
    }

    // TODO: query the elemgraph analysis result
    //       to check if no more offloading appears afterwards.
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
    /* Reset the IO buffer completely. */
    cctx->clear_io_buffers();
}

// vim: ts=8 sts=4 sw=4 et
