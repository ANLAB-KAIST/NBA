#include "common.hh"
#include "log.hh"
#include "offloadtask.hh"
#include "computedevice.hh"
#include "computecontext.hh"
#ifdef USE_CUDA
#include "../engines/cuda/computedevice.hh"
#include "../engines/cuda/computecontext.hh"
#endif
#include "elementgraph.hh"
extern "C" {
#include <rte_common.h>
#include <rte_memcpy.h>
#include <rte_ether.h>
#include <rte_prefetch.h>
}

#include <netinet/ip.h>

using namespace std;
using namespace nba;

static thread_local char dummy_buffer[NBA_MAX_PACKET_SIZE] = {0,};

OffloadTask::OffloadTask()
{
    aligned_elemsizes_h = nullptr;
    aligned_elemsizes_d.ptr = nullptr;
    input_elemsizes_h = nullptr;
    input_elemsizes_d.ptr = nullptr;
    input_buffer_size = 0;
    input_buffer_h = nullptr;
    input_buffer_d.ptr = nullptr;
    output_elemsizes_h = nullptr;
    output_elemsizes_d.ptr = nullptr;
    output_buffer_size = 0;
    output_buffer_h = nullptr;
    output_buffer_d.ptr = nullptr;
    device = nullptr;
    elemgraph = nullptr;
    src_loop = nullptr;
    coproc_ctx = nullptr;
    completion_watcher = nullptr;
    completion_queue   = nullptr;
    assert(offloaded_elements.size() == 0);
    cctx = nullptr;
    num_batches = 0;
    offload_cost = 0;
    offload_start = 0;
}

OffloadTask::~OffloadTask()
{
}

/**
 * First, calculate the required input/output buffer sizes.
 */
size_t OffloadTask::calculate_buffer_sizes()
{
    assert(cctx != NULL);
    OffloadableElement *el = offloaded_elements[0];
    el->get_input_roi(&input_roi);
    el->get_output_roi(&output_roi);
    total_num_pkts = 0;
    for (unsigned i = 0; i < num_batches; i++) {
        total_num_pkts += batches[i]->count;
    }
    print_ratelimit("avg.# pkts sent to GPU", total_num_pkts, 100);
    assert(total_num_pkts > 0);

    cctx->alloc_input_buffer(sizeof(size_t) * total_num_pkts, (void **) &aligned_elemsizes_h, &aligned_elemsizes_d);
    input_buffer_size = 0;

    size_t accum_idx = 0;
    size_t num_valid_pkts = 0;
    switch (input_roi.type) {
    case PARTIAL_PACKET:
    case CUSTOM_INPUT: {
        /* Copy a portion of packets or user-define fixed-size values.
         * We use a fixed-size range (offset, length) here.
         * The buffer is NOT aligned unless the element explicitly
         * specifies the alignment. */
        if (input_roi.align != 0) {
            unsigned aligned_len = RTE_ALIGN_CEIL(input_roi.length, input_roi.align);
            aligned_elemsizes_h[0] = aligned_len;
            input_buffer_size      = aligned_len * total_num_pkts;
            //RTE_LOG(DEBUG, COPROC, "aligned_len: %'u bytes, %lu pkts\n", aligned_len, total_num_pkts);
        } else {
            aligned_elemsizes_h[0] = input_roi.length;
            input_buffer_size      = input_roi.length * total_num_pkts;
            //RTE_LOG(DEBUG, COPROC, "roi_len: %'d bytes, %lu pkts\n", input_roi.length, total_num_pkts);
        }
        for (unsigned i = 0; i < num_batches; i++) {
            PacketBatch *batch = batches[i];
            for (unsigned p = 0; p < batch->count; p++) {
                if (!batch->excluded[p]) num_valid_pkts ++;
            }
        }
        break; }
    case WHOLE_PACKET: {
        /* Copy the whole content of packets.
         * We align the buffer by the cache line size (64 B),
         * or the alignment explicitly set by the element. */
        cctx->alloc_input_buffer(sizeof(size_t) * total_num_pkts, (void **) &input_elemsizes_h, &input_elemsizes_d);
        assert(NULL != input_elemsizes_h);
        size_t align = (input_roi.align == 0) ? 64 : input_roi.align;
        for (unsigned i = 0; i < num_batches; i++) {
            PacketBatch *batch = batches[i];
            for (unsigned p = 0; p < batch->count; p++) {
                unsigned cur_idx = accum_idx + p;
                if (batch->excluded[p]) {
                    input_elemsizes_h[cur_idx]   = 0;
                    aligned_elemsizes_h[cur_idx] = 0;
                } else {
                    unsigned exact_len   = rte_pktmbuf_data_len(batch->packets[p]) - input_roi.offset + input_roi.length;
                    unsigned aligned_len = RTE_ALIGN_CEIL(exact_len, align);
                    input_elemsizes_h[cur_idx]   = exact_len;
                    aligned_elemsizes_h[cur_idx] = aligned_len;
                    input_buffer_size += aligned_len;
                    num_valid_pkts ++;
                }
            }
            accum_idx += batch->count;
        }
        break; }
    default:
        rte_panic("Unsupported input_roi.\n");
        break;
    }
    cctx->total_size_pkts = input_buffer_size;

    output_buffer_size = 0;
    cctx->alloc_output_buffer(sizeof(size_t) * total_num_pkts, (void **) &output_elemsizes_h, &output_elemsizes_d);
    assert(NULL != output_elemsizes_h);
    switch (output_roi.type) {
    case SAME_AS_INPUT:
        rte_memcpy(output_elemsizes_h, aligned_elemsizes_h, sizeof(size_t) * total_num_pkts);
        output_buffer_size = input_buffer_size;
        break;
    case CUSTOM_OUTPUT:
        output_elemsizes_h[0] = output_roi.length;
        output_buffer_size    = output_roi.length * total_num_pkts;
        break;
    default:
        rte_panic("Unsupported output_roi.\n");
        break;
    }
    return num_valid_pkts;
}

/**
 * Allocate host-side buffers and copy necessary data.
 */
void OffloadTask::prepare_host_buffer()
{
    assert(input_buffer_size != 0);
    assert(output_buffer_size != 0);
    cctx->total_num_pkts = total_num_pkts;

    cctx->alloc_input_buffer(input_buffer_size, &input_buffer_h, &input_buffer_d);
    assert(NULL != input_buffer_h);
    assert(NULL != input_buffer_d.ptr);
    cctx->alloc_output_buffer(output_buffer_size, &output_buffer_h, &output_buffer_d);
    assert(NULL != output_buffer_h);
    assert(NULL != output_buffer_d.ptr);

    OffloadableElement *el = offloaded_elements[0];

    /* For PARTIAL_PACKET and CUSTOM_INPUT, input_elemsize is fixed.
     * We use only the first value in the aligned_elemsizes_h array.
     * Note that aligned_elemsizes_h is always set correctly even though
     * the element has not specified explicit alignment. */
    size_t accum_idx = 0;
    size_t input_buffer_offset = 0;
    switch (input_roi.type) {
    case PARTIAL_PACKET:
    case WHOLE_PACKET: {
        /* Copy the speicified region of packet to the input buffer. */
        #define PREFETCH_MAX (4)
        for (unsigned i = 0; i < num_batches; i++) {
            PacketBatch *batch = batches[i];
            #if PREFETCH_MAX
            for (signed p = 0; p < RTE_MIN(PREFETCH_MAX, ((signed)batch->count)); p++)
                if (batch->packets[p] != nullptr)
                    rte_prefetch0(rte_pktmbuf_mtod(batch->packets[p], void*));
            #endif
            for (unsigned p = 0; p < batch->count; p++) {
                unsigned cur_idx = accum_idx + p;
                size_t aligned_elemsz = bitselect<size_t>(input_roi.type == PARTIAL_PACKET,
                                                          aligned_elemsizes_h[0],
                                                          aligned_elemsizes_h[cur_idx]);
                if (batch->excluded[p]) {
                    anno_ptr_array[cur_idx] = nullptr;
                    continue;
                }
                #if PREFETCH_MAX
                if ((signed)p < (signed)batch->count - PREFETCH_MAX && batch->excluded[p + PREFETCH_MAX] == false)
                    rte_prefetch0(rte_pktmbuf_mtod(batch->packets[p + PREFETCH_MAX], void*));
                #endif
                rte_memcpy((char*) input_buffer_h + input_buffer_offset,
                           rte_pktmbuf_mtod(batch->packets[p], char*) + input_roi.offset,
                           aligned_elemsz);
                anno_ptr_array[cur_idx] = &batch->annos[p];
                input_buffer_offset    += aligned_elemsz;
            }
            accum_idx += batch->count;
        }
        #undef PREFETCH_MAX
        break; }
    case CUSTOM_INPUT: {
        /* Call OffloadableElement::preproc() on each packet. */
        for (unsigned i = 0; i < num_batches; i++) {
            PacketBatch *batch = batches[i];
            for (unsigned p = 0; p < batch->count; p++) {
                unsigned cur_idx = accum_idx + p;
                if (batch->excluded[p]) {
                    anno_ptr_array[cur_idx] = nullptr;
                    continue;
                }
                unsigned aligned_elemsz = aligned_elemsizes_h[0];
                el->preproc(input_ports[i],
                            ((char*) input_buffer_h) + input_buffer_offset,
                            batch->packets[p], &batch->annos[p]);
                anno_ptr_array[cur_idx] = &batch->annos[p];
                input_buffer_offset += aligned_elemsz;
            }
            accum_idx += batch->count;
        }
        break; }
    default:
        rte_panic("Unsupported input_roi.\n");
        break;
    }

    /* Calculate resource parameters as default setting. */
    res.num_workitems = cctx->total_num_pkts;
    res.num_threads_per_workgroup = el->get_desired_workgroup_size(cctx->type_name.c_str());
    res.num_workgroups = (cctx->total_num_pkts + res.num_threads_per_workgroup - 1)
                         / res.num_threads_per_workgroup;

    /* Set pointers to various buffers. */
    cctx->set_io_buffers(input_buffer_h, input_buffer_d, input_buffer_size,
                         output_buffer_h, output_buffer_d, output_buffer_size);
    cctx->set_io_buffer_elemsizes(aligned_elemsizes_h, input_elemsizes_d,
                                  sizeof(size_t) * cctx->total_num_pkts,
                                  output_elemsizes_h, output_elemsizes_d,
                                  sizeof(size_t) * cctx->total_num_pkts);

    /* Run element-specific buffer preparation handler. */
    el->prepare_input(cctx, &res, anno_ptr_array);
}

void OffloadTask::copy_buffers_h2d()
{
    assert(total_num_pkts != 0);
    assert(cctx->total_num_pkts != 0);
    assert(input_buffer_size != 0);
    assert(output_buffer_size != 0);

    cctx->clear_checkbits();

    /* Host-to-device copy of input buffer */
    cctx->enqueue_memwrite_op(cctx->get_host_input_buffer_base(),
                              cctx->get_device_input_buffer_base(),
                              0, cctx->get_total_input_buffer_size());

    OffloadableElement *el = offloaded_elements[0];
    handler = el->offload_compute_handlers[cctx->type_name];
    assert(((bool) handler) == true);
}

/**
 * Execute the offload handler and copy back the output data
 * (device-to-host).
 */
void OffloadTask::execute()
{
    /* Call the compute handler of the element.
     * Inside the handler, the element may recalculate the resource
     * parameters. (e.g., IPsec AES use 16B blocks as parallelization
     * unit instead of packets)
     * The element also may add additional memwrite/memread operations. */
    handler(cctx, &res, anno_ptr_array);
}

bool OffloadTask::poll_kernel_finished()
{
    uint8_t *checkbits = cctx->get_host_checkbits();
    if (checkbits == nullptr) return true;
    for (unsigned i = 0; i < res.num_workgroups; i++) {
        if (checkbits[i] == 0)
            return false;
    }
    return true;
}

void OffloadTask::copy_buffers_d2h()
{
    /* Copy the output buffer (device-to-host). */
    cctx->enqueue_memread_op(output_buffer_h, output_buffer_d, 0, output_buffer_size);

    /* Call the completion callback. */
    //cctx->enqueue_event_callback([](ComputeContext *cctx, void *arg) {
    //    OffloadTask *task = (OffloadTask *) arg;
    //    /* Since this callback is called in CUDA-created threads,
    //     * we use an asynchronous event mechanism here to stabilize. */
    //    rte_ring_enqueue(task->coproc_ctx->task_done_queue, (void *) task);
    //    ev_async_send(task->coproc_ctx->loop, task->coproc_ctx->task_done_watcher);
    //}, this);
}

bool OffloadTask::poll_d2h_copy_finished()
{
    return cctx->query();
}

void OffloadTask::notify_completion()
{
    /* Notify the computation thread. */
    assert(0 == rte_ring_sp_enqueue(completion_queue, (void *) this));
    ev_async_send(src_loop, completion_watcher);
}

void OffloadTask::postproc()
{
    unsigned accum_idx = 0;
    int cur_idx;
    OffloadableElement *el = offloaded_elements[0];

    switch (output_roi.type) {
    case SAME_AS_INPUT: {
        /* Update the packets and run postprocessing. */
        size_t output_buffer_offset = 0;
        for (unsigned i = 0; i < num_batches; i++) {
            PacketBatch *batch = batches[i];
            for (unsigned p = 0; p < batch->count; p++) {
                if (batch->excluded[p]) continue;
                cur_idx = accum_idx + p;
                size_t elemsz = bitselect<size_t>(output_roi.type == SAME_AS_INPUT && input_roi.type == WHOLE_PACKET,
                                                  output_elemsizes_h[cur_idx],
                                                  output_elemsizes_h[0]);
                rte_memcpy(rte_pktmbuf_mtod(batch->packets[p], char*) + output_roi.offset,
                            (char*) output_buffer_h + output_buffer_offset,
                            elemsz);
                batch->results[p] = el->postproc(input_ports[i], NULL,
                                                batch->packets[p], &batch->annos[p]);
                batch->excluded[p] = (batch->results[p] == DROP);
                output_buffer_offset += elemsz;
            }
            accum_idx += batch->count;
            batch->has_results = true;
        }
        break; }
    case CUSTOM_OUTPUT: {
        /* Run postporcessing only. */
        size_t output_buffer_offset = 0;
        for (unsigned i = 0; i < num_batches; i++) {
            PacketBatch *batch = batches[i];
            assert(accum_idx + batch->count <= el->ctx->num_coproc_ppdepth * NBA_MAX_COMPBATCH_SIZE);
            for (unsigned p = 0; p < batch->count; p++) {
                if (batch->excluded[p]) continue;
                cur_idx = accum_idx + p;
                unsigned elemsz = output_elemsizes_h[0];
                batch->results[p] = el->postproc(input_ports[i],
                                                (char*) output_buffer_h + output_buffer_offset,
                                                batch->packets[p], &batch->annos[p]);
                batch->excluded[p] = (batch->results[p] == DROP);
                output_buffer_offset += elemsz;
            }
            accum_idx += batch->count;
            batch->has_results = true;
        }
        break; }
    default:
        rte_panic("Unsupported output_roi.\n");
        break;
    }
}

// vim: ts=8 sts=4 sw=4 et
