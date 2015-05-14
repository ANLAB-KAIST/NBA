#include "elementgraph.hh"
#include "config.hh"
#include "log.hh"
#include "io.hh"
#include "computedevice.hh"
#include "computecontext.hh"
#include "loadbalancer.hh"

#include <rte_cycles.h>
#include <rte_memory.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>
#include <rte_branch_prediction.h>
#ifdef USE_NVPROF
#include <nvToolsExt.h>
#endif

#include <cassert>


static const uint64_t BRANCH_TRUNC_LIMIT = (1<<24);

using namespace std;
using namespace nba;

ElementGraph::ElementGraph(comp_thread_context *ctx)
    : elements(128, ctx->loc.node_id), sched_elements(16, ctx->loc.node_id),
      queue(2048, ctx->loc.node_id), delayed_batches(2048, ctx->loc.node_id)
{
    this->ctx = ctx;
    input_elem = nullptr;
    assert(0 == rte_malloc_validate(ctx, NULL));
    /* IMPORTANT: ready_tasks must be larger than task_pool. */
    for (int i = 0; i < NBA_MAX_COPROCESSOR_TYPES; i++)
        ready_tasks[i].init(256, ctx->loc.node_id);
}

void ElementGraph::flush_offloaded_tasks()
{
    if (unlikely(ctx->io_ctx->loop_broken))
        return;
    uint64_t len_ready_tasks = ready_tasks[0].size();
    print_ratelimit("# ready tasks", len_ready_tasks, 10000);

    for (int dev_idx = 0; dev_idx < NBA_MAX_COPROCESSOR_TYPES; dev_idx++) {
        // TODO: now it's possible to merge multiple tasks to increase batch size!
        while (!ready_tasks[dev_idx].empty()) {
            OffloadTask *task = ready_tasks[dev_idx].front();
            ready_tasks[dev_idx].pop_front();
            //task->offload_start = rte_rdtsc();

            /* Start offloading! */
            // TODO: create multiple cctx_list and access them via dev_idx for hetero-device systems.
            //if (!ctx->cctx_list.empty()) {
            ComputeContext *cctx = ctx->cctx_list.front();
                //ctx->cctx_list.pop_front();
            if (cctx->state == ComputeContext::READY) {

                /* Grab a compute context. */
                assert(cctx != nullptr);
                assert(cctx->state == ComputeContext::READY);
                #ifdef USE_NVPROF
                nvtxRangePush("offl_prepare");
                #endif

                /* Prepare to offload. */
                cctx->state = ComputeContext::PREPARING;
                cctx->currently_running_task = task;
                // FIXME: dedicate a single cctx to each computation thread
                //        (COPROC_CTX_PER_COMPTHREAD 설정이 1이면 괜찮지만 아예 1로 고정할 것.)
                task->cctx = cctx;

                int datablock_ids[NBA_MAX_DATABLOCKS];
                size_t num_db_used = task->elem->get_used_datablocks(datablock_ids);
                for (unsigned k = 0; k < num_db_used; k++) {
                    int dbid = datablock_ids[k];
                    task->datablocks.push_back(dbid);
                    task->dbid_h2d[dbid] = k;
                }

                //size_t total_num_pkts = 0;
                for (PacketBatch *batch : task->batches) {
                    //total_num_pkts = batch->count;
                    if (batch->datablock_states == nullptr)
                        assert(0 == rte_mempool_get(ctx->dbstate_pool, (void **) &batch->datablock_states));
                    //assert(task->offload_start);
                    //task->offload_cost += (rte_rdtsc() - task->offload_start);
                    task->offload_start = 0;
                }
                //print_ratelimit("avg.# pkts sent to GPU", total_num_pkts, 100);
                //assert(total_num_pkts > 0);

                task->prepare_read_buffer();
                task->prepare_write_buffer();

                /* Until this point, no device operations (e.g., CUDA API calls) has been executed. */
                /* Enqueue the offload task. */
                assert(0 == rte_ring_enqueue(ctx->offload_input_queues[dev_idx], (void*) task));
                ev_async_send(ctx->coproc_ctx->loop, ctx->offload_devices->at(dev_idx)->input_watcher);
                if (ctx->inspector) ctx->inspector->dev_sent_batch_count[0] += task->batches.size();
                #ifdef USE_NVPROF
                nvtxRangePop();
                #endif

            } else {

                /* There is no available compute contexts.
                 * Delay the current offloading task and break. */
                ready_tasks[dev_idx].push_back(task);
                break;

            }
        }
    }
}

void ElementGraph::flush_delayed_batches()
{
    uint64_t prev_gen = 0;
    uint64_t len_delayed_batches = delayed_batches.size();
    print_ratelimit("# delayed batches", len_delayed_batches, 10000);

    while (!delayed_batches.empty() && !ctx->io_ctx->loop_broken) {
        PacketBatch *batch = delayed_batches.front();
        delayed_batches.pop_front();
        if(batch->delay_start > 0)
        {
            batch->compute_time += (rte_rdtsc()-batch->delay_start);
            batch->delay_start = 0;
        }

        /* It must have the associated element where this batch is delayed. */
        assert(batch->element != nullptr);

        /* Re-run the element graph from that element. */
        run(batch, batch->element, batch->input_port);
    }
}

void ElementGraph::free_batch(PacketBatch *batch, bool free_pkts)
{
    if (free_pkts) {
        for (Packet *pkt = batch->first_packet; pkt != nullptr; pkt = pkt->next) {
            rte_ring_enqueue(ctx->io_ctx->drop_queue, (void *) pkt->base);  // only for separate comp/io threads
        }
    }
    rte_mempool_put(ctx->batch_pool, (void *) batch);
}

void ElementGraph::run(PacketBatch *batch, Element *start_elem, int input_port)
{
    Element *el = start_elem;
    assert(el != nullptr);
    assert(ctx->io_ctx != nullptr);

    batch->element = el;
    batch->input_port = input_port;
    batch->generation ++;
    queue.push_back(batch);

    /*
     * We have two cases of batch handling:
     *   - single output: enqueue the given batch as it is without
     *     copying or memory pool allocation.
     *   - multiple outputs: make copy of batches (but not packets) and
     *     enqueue them as later job.
     */

    /* When the queue becomes empty, the processing path started from
     * the start_elem is finished.  The unit of a job is an element. */
    while (!queue.empty() && !ctx->io_ctx->loop_broken) {
        PacketBatch *batch = queue.front();
        queue.pop_front();

        Element *current_elem = batch->element;
        int input_port = batch->input_port;
        int batch_disposition = CONTINUE_TO_PROCESS;
        int64_t lb_decision = anno_get(&batch->banno, NBA_BANNO_LB_DECISION);
        uint64_t _cpu_start = rte_rdtsc();

        /* Check if we can and should offload. */
        if (!batch->has_results) {
            if (current_elem->get_type() & ELEMTYPE_OFFLOADABLE) {
                OffloadableElement *offloadable = dynamic_cast<OffloadableElement*>(current_elem);
                assert(offloadable != nullptr);
                if (lb_decision != -1) {

                    /* Get or initialize the task object.
                     * This step is always executed for every input batch
                     * passing every offloadable element. */
                    int dev_idx = lb_decision;
                    OffloadTask *task = nullptr;
                    if (offloadable->tasks[dev_idx] == nullptr) {
                        #ifdef USE_NVPROF
                        nvtxRangePush("accum_batch");
                        #endif
                        /* We assume: task pool size >= task input queue length */
                        //printf("current cuda task qlen: %lu\n", ready_tasks[dev_idx].size());
                        int ret = rte_mempool_get(ctx->task_pool, (void **) &task);
                        if (ret == -ENOENT) {
                            if (!ctx->io_ctx->loop_broken)
                                ev_run(ctx->io_ctx->loop, EVRUN_NOWAIT);
                            /* Keep the current batch for later processing. */
                            assert(batch->delay_start == 0);
                            batch->delay_start = rte_rdtsc();
                            delayed_batches.push_back(batch);
                            continue;
                        }
                        new (task) OffloadTask();
                        task->src_loop = ctx->loop;
                        task->comp_ctx = ctx;
                        task->completion_queue = ctx->task_completion_queue;
                        task->completion_watcher = ctx->task_completion_watcher;
                        task->elemgraph = this;
                        task->local_dev_idx = dev_idx;
                        //task->device = ctx->offload_devices->at(dev_idx);
                        //assert(task->device != nullptr);
                        task->elem = offloadable;

                        offloadable->tasks[dev_idx] = task;
                    } else
                        task = offloadable->tasks[dev_idx];
                    assert(task != nullptr);

                    /* Add the current batch if the task batch is not full. */
                    if (task->batches.size() < ctx->num_coproc_ppdepth) {
                        task->batches.push_back(batch);
                        task->input_ports.push_back(input_port);
                        #ifdef USE_NVPROF
                        nvtxMarkA("add_batch");
                        #endif
                    } else {
                        /* We have no room for batch in the preparing task.
                         * Keep the current batch for later processing. */
                        assert(batch->delay_start == 0);
                        batch->delay_start = rte_rdtsc();
                        delayed_batches.push_back(batch);
                        continue;
                    }

                    /* Start offloading when one or more following conditions are met:
                     *  1. The task batch size has reached the limit (num_coproc_ppdepth).
                     *     This is the ideal case that pipelining happens and GPU utilization is high.
                     *  2. The load balancer has changed its decision to CPU.
                     *     We need to flush the pending offload tasks.
                     *  3. The time elapsed since the first packet is greater than
                     *     10 x avg.task completion time.
                     */
                    assert(task->batches.size() > 0);
                    uint64_t now = _cpu_start;
                    if (task->batches.size() == ctx->num_coproc_ppdepth
                        //|| (ctx->load_balancer != nullptr && ctx->load_balancer->is_changed_to_cpu)
                        //|| (now - task->batches[0]->recv_timestamp) / (float) rte_get_tsc_hz() 
                        //    > ctx->inspector->avg_task_completion_sec[dev_idx] * 10
                        )//|| (ctx->io_ctx->mode == IO_EMUL && !ctx->stop_task_batching))
                    {
                        //printf("avg task completion time: %.6f sec\n", ctx->inspector->avg_task_completion_sec[dev_idx]);

                        offloadable->tasks[dev_idx] = nullptr;  // Let the element be able to take next pkts/batches.
                        task->begin_timestamp = now;
                        task->offload_start = _cpu_start;

                        ready_tasks[dev_idx].push_back(task);
                        #ifdef USE_NVPROF
                        nvtxRangePop();
                        #endif
                        flush_offloaded_tasks();
                    }

                    /* At this point, the batch is already consumed to the task
                     * or delayed. */

                    //if (ctx->load_balancer) ctx->load_balancer->is_changed_to_cpu = false;
                    continue;

                } else {
                    /* If not offloaded, run the element's CPU-version handler. */

                    batch_disposition = current_elem->_process_batch(input_port, batch);
                    double _cpu_end = rte_rdtsc();
                    batch->compute_time += (_cpu_end - _cpu_start);
                }
            } else {
                /* If not offloadable, run the element's CPU-version handler. */
                batch_disposition = current_elem->_process_batch(input_port, batch);
            }
        }

        /* If the element was per-batch and it said it will keep the batch,
         * we do not have to perform batch-split operations below. */
        if (batch_disposition == KEPT_BY_ELEMENT)
            continue;

        /* When offloading is complete, processing of the resultant batches begins here.
         * (ref: enqueue_postproc_batch) */

        /* Here, we should have the results no matter what happened before. */
        assert(batch->has_results == true);

        //assert(current_elem->num_max_outputs <= num_max_outputs || current_elem->num_max_outputs == -1);
        int num_outputs = current_elem->next_elems.size();

        if (num_outputs == 0) {

            /* If no outputs are connected, drop all packets. */
            if (ctx->inspector) ctx->inspector->drop_pkt_count += batch->count;
            free_batch(batch);
            continue;

        } else if (num_outputs == 1) {

            /* With the single output, we don't need to allocate new
             * batches.  Just reuse the given one. */
            if (0 == (current_elem->get_type() & ELEMTYPE_PER_BATCH)) {
                Packet *prev_pkt = nullptr;
                for (Packet *pkt = batch->first_packet; pkt != nullptr; pkt = pkt->next) {
                    int o = pkt->result;
                    switch (o) {
                    case DROP:
                        if (ctx->inspector) ctx->inspector->drop_pkt_count += batch->count;
                        rte_ring_enqueue(ctx->io_ctx->drop_queue, (void *) pkt->base);
                        if (prev_pkt == nullptr)
                            batch->first_packet = pkt->next;
                        else
                            prev_pkt->next = pkt->next;
                        batch->count--;
                        break;
                    case PENDING:
                        // Remove from the batch, but don't free..
                        // They are stored in io_thread_ctx::pended_pkt_queue.
                        if (prev_pkt == nullptr)
                            batch->first_packet = pkt->next;
                        else
                            prev_pkt->next = pkt->next;
                        batch->count--;
                        break;
                    case SLOWPATH:
                        rte_panic("SLOWPATH is not supported yet. (element: %s)\n", current_elem->class_name());
                        break;
                    case 0:
                        // pass
                        break;
                    default:
                        rte_panic("Invalid packet disposition value. (element: %s, value: %d)\n", current_elem->class_name(), o);
                    }
                    prev_pkt = pkt;
                }
            }
            if (current_elem->next_elems[0]->get_type() & ELEMTYPE_OUTPUT) {
                /* We are at the end leaf of the pipeline.
                 * Inidicate free of the original batch. */
                if (ctx->inspector) {
                    ctx->inspector->tx_batch_count ++;;
                    ctx->inspector->tx_pkt_count += batch->count;
                }
                //printf("txtx? (%u)\n", rte_mempool_count(ctx->io_ctx->emul_rx_packet_pool));
                io_tx_batch(ctx->io_ctx, batch);
                //printf("txtx! (%u)\n", rte_mempool_count(ctx->io_ctx->emul_rx_packet_pool));
                free_batch(batch, false);
                continue;
            } else {
                /* Recurse into the next element, reusing the batch. */
                Element *next_el = current_elem->next_elems[0];
                int next_input_port = current_elem->next_connected_inputs[0];

                batch->element = next_el;
                batch->input_port = next_input_port;
                batch->has_results = false;
                queue.push_back(batch);
                continue;
            }

        } else { /* num_outputs > 1 */

            PacketBatch *out_batches[num_max_outputs];
            Packet *out_last_packets[num_max_outputs];
            // TODO: implement per-batch handling for branches
#ifndef NBA_DISABLE_BRANCH_PREDICTION
#ifndef NBA_BRANCH_PREDICTION_ALWAYS
            /* use branch prediction when miss < total / 4. */
            if ((current_elem->branch_total >> 2) > (current_elem->branch_miss))
#endif
            {
                /* With multiple outputs, make copy of batches.
                 * This does not copy the content of packets but the
                 * pointers to packets. */
                int predicted_output = 0; //TODO set to right prediction
                uint64_t current_max = 0;
                for (unsigned k = 0; k < current_elem->next_elems.size(); k++) {
                    if (current_max < current_elem->branch_count[k]) {
                        current_max = current_elem->branch_count[k];
                        predicted_output = k;
                    }
                }

                memset(out_batches, 0, num_max_outputs * sizeof(PacketBatch *));
                memset(out_last_packets, 0, num_max_outputs * sizeof(Packet *));
                out_batches[predicted_output] = batch;

                /* Classify packets into copy-batches. */
                Packet *prev_pkt = nullptr;
                Packet *pkt = nullptr;
                for (pkt = batch->first_packet; pkt != nullptr; pkt = pkt->next) {
                    int o = pkt->result;
                    assert(o < num_outputs);

                    /* Prediction mismatch! */
                    if (unlikely(o != predicted_output)) {
                        if (o >= 0) {
                            if (!out_batches[o]) {
                                /* out_batch is not allocated yet... */
                                while (rte_mempool_get(ctx->batch_pool, (void**)(out_batches + o)) == -ENOENT
                                       && !ctx->io_ctx->loop_broken) {
                                    ev_run(ctx->io_ctx->loop, EVRUN_NOWAIT);
                                    flush_delayed_batches();
                                    flush_offloaded_tasks();
                                }
                                new(out_batches[o]) PacketBatch();
                                anno_set(&out_batches[o]->banno, NBA_BANNO_LB_DECISION, lb_decision);
                                out_batches[o]->recv_timestamp = batch->recv_timestamp;
                            }
                            /* Append the packet to the output batch. */
                            int cnt = out_batches[o]->count++;
                            if (out_last_packets[o] == nullptr) {
                                out_batches[o]->first_packet = pkt;
                                out_last_packets[o] = pkt;
                            } else {
                                out_last_packets[o]->next = pkt;
                                out_last_packets[o] = pkt;
                            }
                            /* Exclude it from the batch. */
                            if (prev_pkt == nullptr)
                                batch->first_packet = pkt->next;
                            else
                                prev_pkt->next = pkt->next;
                            batch->count--;
                        } else if (o == DROP) {
                            /* Let the IO loop free the packet. */
                            if (ctx->inspector) ctx->inspector->drop_pkt_count += 1;
                            rte_ring_enqueue(ctx->io_ctx->drop_queue, (void *) pkt->base);
                            /* Exclude it from the batch. */
                            if (prev_pkt == nullptr)
                                batch->first_packet = pkt->next;
                            else
                                prev_pkt->next = pkt->next;
                            batch->count--;
                        } else if (o == PENDING) {
                            /* The packet is stored in io_thread_ctx::pended_pkt_queue. */
                            /* Exclude it from the batch. */
                            if (prev_pkt == nullptr)
                                batch->first_packet = pkt->next;
                            else
                                prev_pkt->next = pkt->next;
                            batch->count--;
                        } else if (o == SLOWPATH) {
                            assert(0); // Not implemented yet.
                        } else {
                            rte_panic("Invalid packet disposition value. (element: %s, value: %d)\n", current_elem->class_name(), o);
                        }
                        current_elem->branch_miss++;
                    }
                    current_elem->branch_total++;
                    current_elem->branch_count[o]++;
                    prev_pkt = pkt;
                }
                /* Nullify the tails. */
                for (unsigned o = 0; o < num_max_outputs; o++)
                    if (out_last_packets[o] != nullptr) out_last_packets[o]->next = nullptr;

                if (current_elem->branch_total & BRANCH_TRUNC_LIMIT) {
                    //double percentage = ((double)(current_elem->branch_total-current_elem->branch_miss) / (double)current_elem->branch_total);
                    //printf("%s: prediction: %f\n", current_elem->class_name(), percentage);
                    current_elem->branch_miss = current_elem->branch_total = 0;
                    for (unsigned k = 0; k < current_elem->next_elems.size(); k++)
                        current_elem->branch_count[k] = current_elem->branch_count[k] >> 1;
                }

                /* Recurse into the element subgraph starting from each
                 * output port using copy-batches. */
                for (int o = 0; o < num_outputs; o++) {
                    if (out_batches[o] && out_batches[o]->count > 0) {
                        assert(current_elem->next_elems[o] != NULL);
                        if (current_elem->next_elems[o]->get_type() & ELEMTYPE_OUTPUT) {

                            if (ctx->inspector) {
                                ctx->inspector->tx_batch_count ++;
                                ctx->inspector->tx_pkt_count += out_batches[o]->count;
                            }

                            /* We are at the end leaf of the pipeline. */
                            io_tx_batch(ctx->io_ctx, out_batches[o]);
                            free_batch(out_batches[o], false);

                        } else {

                            Element *next_el = current_elem->next_elems[o];
                            int next_input_port = current_elem->next_connected_inputs[o];

                            out_batches[o]->element = next_el;
                            out_batches[o]->input_port = next_input_port;
                            out_batches[o]->has_results = false;

                            /* Push at the beginning of the job queue (DFS).
                             * If we insert at the end, it becomes BFS. */
                            queue.push_back(out_batches[o]);
                        }
                    } else {
                        /* This batch is unused! */
                        if (out_batches[o])
                            free_batch(out_batches[o]);
                    }
                }
                continue;
            }
#ifndef NBA_BRANCH_PREDICTION_ALWAYS
            else
#endif
#endif
#ifndef NBA_BRANCH_PREDICTION_ALWAYS
            {
                while (rte_mempool_get_bulk(ctx->batch_pool, (void **) out_batches, num_outputs) == -ENOENT
                       && !ctx->io_ctx->loop_broken) {
                    ev_run(ctx->io_ctx->loop, EVRUN_NOWAIT);
                    flush_delayed_batches();
                    flush_offloaded_tasks();
                }

                /* Initialize copy-batches. */
                for (int o = 0; o < num_outputs; o++) {
                    new (out_batches[o]) PacketBatch();
                    anno_set(&out_batches[o]->banno, NBA_BANNO_LB_DECISION, lb_decision);
                    out_batches[o]->recv_timestamp = batch->recv_timestamp;
                }
                memset(out_last_packets, 0, num_max_outputs * sizeof(Packet *));

                /* Classify packets into copy-batches. */
                Packet *prev_pkt = nullptr;
                Packet *pkt = nullptr;
                for (pkt = batch->first_packet; pkt != nullptr; pkt = pkt->next) {
                    int o = pkt->result;
                    assert(o < num_outputs);

                    if (o >= 0) {
                        int cnt = out_batches[o]->count ++;
                        /* Append the packet to the output batch. */
                        if (out_last_packets[o] == nullptr) {
                            out_batches[o]->first_packet = pkt;
                            out_last_packets[o] = pkt;
                        } else {
                            out_last_packets[o]->next = pkt;
                            out_last_packets[o] = pkt;
                        }
                    } else if (o == DROP) {
                        if (ctx->inspector) ctx->inspector->drop_pkt_count += batch->count;
                        rte_ring_enqueue(ctx->io_ctx->drop_queue, (void *) pkt->base);
                    } else if (o == PENDING) {
                        // Remove from PacketBatch, but don't free..
                        // They are stored in io_thread_ctx::pended_pkt_queue.
                    } else if (o == SLOWPATH) {
                        assert(0);
                    } else {
                        rte_panic("Invalid packet disposition value. (element: %s, value: %d)\n", current_elem->class_name(), o);
                    }
                }
                /* Nullify the tails. */
                for (unsigned o = 0; o < num_max_outputs; o++)
                    if (out_last_packets[o] != nullptr) out_last_packets[o]->next = nullptr;
                batch->first_packet = nullptr;

                /* Recurse into the element subgraph starting from each
                 * output port using copy-batches. */
                for (int o = 0; o < num_outputs; o++) {
                    if (out_batches[o]->count > 0) {
                        assert(current_elem->next_elems[o] != NULL);
                        if (current_elem->next_elems[o]->get_type() & ELEMTYPE_OUTPUT) {

                            if (ctx->inspector) {
                                ctx->inspector->tx_batch_count ++;
                                ctx->inspector->tx_pkt_count += batch->count;
                            }

                            /* We are at the end leaf of the pipeline. */
                            io_tx_batch(ctx->io_ctx, out_batches[o]);
                            free_batch(out_batches[o], false);

                        } else {

                            Element *next_el = current_elem->next_elems[o];
                            int next_input_port = current_elem->next_connected_inputs[o];

                            out_batches[o]->element = next_el;
                            out_batches[o]->input_port = next_input_port;
                            out_batches[o]->has_results = false;

                            /* Push at the beginning of the job queue (DFS).
                             * If we insert at the end, it becomes BFS. */
                            queue.push_back(out_batches[o]);
                        }
                    } else {
                        /* This batch is unused! */
                        free_batch(out_batches[o]);
                    }
                }

                /* With multiple outputs (branches happened), we have made
                 * copy-batches and the parent should free its batch. */
                free_batch(batch);
                continue;
            }
#endif
        }
    }
    return;
}

void ElementGraph::enqueue_postproc_batch(PacketBatch *batch, Element *offloaded_elem, int input_port)
{
    assert(offloaded_elem != nullptr);
    assert(batch->has_results == true);
    batch->element = offloaded_elem;
    batch->input_port = input_port;
    delayed_batches.push_back(batch);
}

bool ElementGraph::check_datablock_reuse(Element *offloaded_elem, int datablock_id)
{
    //bool is_offloadable = ((offloaded_elem->next_elems[0]->get_type() & ELEMTYPE_OFFLOADABLE) != 0);
    //int used_dbids[NBA_MAX_DATABLOCKS];
    //if (is_offloadable) {
    //    OffloadableElement *oelem = dynamic_cast<OffloadableElement*> (offloaded_elem);
    //    assert(oelem != nullptr);
    //    size_t n = oelem->get_used_datablocks(used_dbids);
    //    for (unsigned i = 0; i < n; i++)
    //        if (used_dbids[i] == datablock_id)
    //            return true;
    //}
    return false;
}

bool ElementGraph::check_next_offloadable(Element *offloaded_elem)
{
    // FIXME: generalize for branched offloaded_elem
    return ((offloaded_elem->next_elems[0]->get_type() & ELEMTYPE_OFFLOADABLE) != 0);
}

int ElementGraph::add_element(Element *new_elem)
{
    new_elem->update_port_count();
    new_elem->ctx = this->ctx;
    elements.push_back(new_elem);

    if (new_elem->get_type() & ELEMTYPE_SCHEDULABLE) {
        auto selem = dynamic_cast<SchedulableElement*> (new_elem);
        assert(selem != nullptr);
        sched_elements.push_back(selem);
    }
    if (new_elem->get_type() & ELEMTYPE_INPUT) {
        assert(input_elem == nullptr);
        input_elem = dynamic_cast<SchedulableElement*> (new_elem);
        assert(input_elem != nullptr);
    }
    return 0;
}

int ElementGraph::link_element(Element *to_elem, int input_port,
                               Element *from_elem, int output_port)
{
    if (from_elem != NULL) {
        bool found = false;
        for (unsigned i = 0; i < elements.size(); i++) {
            if (from_elem == elements[i]) {
                found = true;
                break;
            }
        }
        assert(found);
        assert(output_port < from_elem->num_max_outputs || from_elem->num_max_outputs == -1);
        assert(input_port < to_elem->num_max_inputs);
        assert(from_elem->next_elems.size() == (unsigned) output_port);
        from_elem->next_elems.push_back(to_elem);
        assert(from_elem->next_connected_inputs.size() == (unsigned) output_port);
        from_elem->next_connected_inputs.push_back(input_port);
    }
    return 0;
}

SchedulableElement *ElementGraph::get_entry_point(int entry_point_idx)
{
    // TODO: implement multiple packet entry points.
    assert(input_elem != nullptr);
    return input_elem;
}

int ElementGraph::validate()
{
    // TODO: implement
    return 0;
}

const FixedRing<SchedulableElement*, nullptr>& ElementGraph::get_schedulable_elements() const
{
    return sched_elements;
}

const FixedRing<Element*, nullptr>& ElementGraph::get_elements() const
{
    return elements;
}

// vim: ts=8 sts=4 sw=4 et
