#include <nba/framework/config.hh>
#include <nba/framework/elementgraph.hh>
#include <nba/framework/logging.hh>
#include <nba/framework/io.hh>
#include <nba/framework/computedevice.hh>
#include <nba/framework/computecontext.hh>
#include <nba/framework/loadbalancer.hh>
#include <nba/framework/task.hh>
#include <nba/framework/offloadtask.hh>
#include <nba/element/packetbatch.hh>
#include <nba/core/logging.hh>
#include <nba/core/enumerate.hh>
#include <nba/core/timing.hh>
#include <cassert>
#include <rte_cycles.h>
#include <rte_memory.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>
#include <rte_branch_prediction.h>
#include <rte_hash.h>
#include <rte_hash_crc.h>
#ifdef USE_NVPROF
#include <nvToolsExt.h>
#endif


static const uint64_t BRANCH_TRUNC_LIMIT = (1<<24);

using namespace std;
using namespace nba;

ElementGraph::ElementGraph(comp_thread_context *ctx)
    : elements(128, ctx->loc.node_id),
      sched_elements(16, ctx->loc.node_id),
      offl_elements(16, ctx->loc.node_id),
      queue(2048, ctx->loc.node_id)
{
    const size_t ready_task_qlen = 256;
    this->ctx = ctx;
    input_elem = nullptr;
    assert(0 == rte_malloc_validate(ctx, NULL));

#ifdef NBA_REUSE_DATABLOCKS
    struct rte_hash_parameters hparams;
    char namebuf[RTE_HASH_NAMESIZE];
    snprintf(namebuf, RTE_HASH_NAMESIZE, "elemgraph@%u.%u:offl_actions",
             ctx->loc.node_id, ctx->loc.local_thread_idx);
    hparams.name = namebuf;
    hparams.entries = 64;
    hparams.key_len = sizeof(struct offload_action_key);
    hparams.hash_func = rte_hash_crc;
    hparams.hash_func_init_val = 0;
    hparams.socket_id = ctx->loc.node_id;
    offl_actions = rte_hash_create(&hparams);
    assert(offl_actions != nullptr);
#else
    offl_actions = nullptr;
#endif
}

void ElementGraph::send_offload_task_to_device(OffloadTask *task)
{
    if (unlikely(ctx->io_ctx->loop_broken))
        return;

    /* Start offloading! */
    // TODO: create multiple cctx_list and access them via dev_idx for hetero-device systems.
    const int dev_idx = 0;
    ComputeContext *cctx = ctx->cctx_list->front();
    assert(cctx != nullptr);
    #ifdef USE_NVPROF
    nvtxRangePush("offl_prepare");
    #endif
    task->cctx = cctx;

    /* Prepare to offload. */
    if (task->state < TASK_PREPARED) {
        /* In the GPU side, datablocks argument has only used
         * datablocks in the beginning of the array (not sparsely). */
        int datablock_ids[NBA_MAX_DATABLOCKS];
        size_t num_db_used = task->elem->get_used_datablocks(datablock_ids);
        for (unsigned k = 0; k < num_db_used; k++) {
            int dbid = datablock_ids[k];
            task->datablocks.push_back(dbid);
            task->dbid_h2d[dbid] = k;
        }

        /* As we reuse tasks between subsequent offloadables
         * and only does in linear groups of elements,
         * it is okay to check only the first batch. */
        size_t num_batches = task->batches.size();
        if (task->batches[0]->datablock_states == nullptr) {
            struct datablock_tracker *dbstates[num_batches];
            assert(0 == rte_mempool_get_bulk(ctx->dbstate_pool,
                                             (void **) &dbstates,
                                             num_batches));
            for (auto&& p : enumerate(task->batches))
                (p.second)->datablock_states = dbstates[p.first];
        }

        /* Allocate the host-device IO buffer pool. */
        while (task->io_base == INVALID_IO_BASE) {
            task->io_base = cctx->alloc_io_base();
            if (unlikely(ctx->io_ctx->loop_broken)) return;
            if (task->io_base == INVALID_IO_BASE) {
                /* If not available now, wait. */
                ev_run(ctx->io_ctx->loop, 0);
            }
        }

        /* Calculate required buffer sizes, allocate them, and initialize them.
         * The mother buffer is statically allocated on start-up and here we
         * reserve regions inside it. */
        task->prepare_read_buffer();
        task->prepare_write_buffer();
        task->state = TASK_PREPARED;
    }

    /* Send the offload task to device thread. */
    assert(task->state == TASK_PREPARED);
    int ret = rte_ring_enqueue(ctx->offload_input_queues[dev_idx], (void*) task);
    if (ret == -ENOBUFS) {
        /* The input queue is full.  Delay the task. */
        enqueue_offload_task(task, task->tracker.element, task->tracker.input_port);
    } else {
        /* It may return -EDQUOT, but here we ignore this HWM signal.
         * Even for that case, the task is enqueued successfully. */
        ev_async_send(ctx->coproc_ctx->loop, ctx->offload_devices->at(dev_idx)->input_watcher);
        if (ctx->inspector) ctx->inspector->dev_sent_batch_count[0] += task->batches.size();
    }
    #ifdef USE_NVPROF
    nvtxRangePop();
    #endif
    return;
}

void ElementGraph::free_batch(PacketBatch *batch, bool free_pkts)
{
    if (free_pkts) {
        assert(0 == rte_ring_enqueue_bulk(ctx->io_ctx->drop_queue,
                                          (void **) &batch->packets[0],
                                          batch->count));
        #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
        assert(0 == rte_ring_enqueue_bulk(ctx->io_ctx->drop_queue,
                                          (void **) &batch->packets[batch->count],
                                          batch->drop_count));
        #endif
    }
    rte_mempool_put(ctx->batch_pool, (void *) batch);
    /* Make any blocking call to ev_run() to break. */
    ev_break(ctx->io_ctx->loop, EVBREAK_ALL);
}

void ElementGraph::scan_schedulable_elements(uint64_t loop_count)
{
    uint64_t now = 0;
    if ((loop_count & 0x3ff) == 0)
        now = get_usec();
    for (SchedulableElement *selem : sched_elements) {
        /* FromInput is handled by feed_input() invoked by comp_process_batch(). */
        if (0 == (selem->get_type() & ELEMTYPE_INPUT)) {
            PacketBatch *next_batch = nullptr;
            if (now > 0 && selem->_last_delay != 0 && now >= selem->_last_call_ts + selem->_last_delay) {
                selem->dispatch(loop_count, next_batch, selem->_last_delay);
                selem->_last_call_ts = now;
            }
            if (selem->_last_delay == 0) {
                selem->dispatch(loop_count, next_batch, selem->_last_delay);
            }
            /* Try to "drain" internally stored batches. */
            while (next_batch != nullptr) {
                next_batch->tracker.has_results = true; // skip processing
                enqueue_batch(next_batch, selem, 0);
                selem->dispatch(loop_count, next_batch, selem->_last_delay);
            };
        } /* endif(!ELEMTYPE_INPUT) */
    } /* endfor(selems) */
}

void ElementGraph::scan_offloadable_elements(uint64_t loop_count)
{
    PacketBatch *next_batch = nullptr;
    for (OffloadableElement *oelem : offl_elements) {
        do {
            next_batch = nullptr;
            oelem->dispatch(loop_count, next_batch, oelem->_last_delay);
            if (next_batch != nullptr) {
                next_batch->tracker.has_results = true;
                enqueue_batch(next_batch, oelem, 0);
            }
        } while (next_batch != nullptr);
    } /* endfor(oelems) */
}

void ElementGraph::feed_input(int entry_point_idx, PacketBatch *batch, uint64_t loop_count)
{
    uint64_t next_delay = 0; /* unused but required to pass as reference */
    // TODO: implement multiple packet entry points.
    SchedulableElement *input_elem = this->input_elem;
    PacketBatch *next_batch = nullptr;
    assert(0 != (input_elem->get_type() & ELEMTYPE_INPUT));
    ctx->input_batch = batch;
    input_elem->dispatch(loop_count, next_batch, next_delay);
    if (next_batch == nullptr) {
        free_batch(batch);
    } else {
        assert(next_batch == batch);
        next_batch->tracker.has_results = true; // skip processing
        enqueue_batch(next_batch, input_elem, 0);
    }
}

void ElementGraph::enqueue_batch(PacketBatch *batch, Element *start_elem, int input_port)
{
    assert(start_elem != nullptr);
    batch->tracker.element = start_elem;
    batch->tracker.input_port = input_port;
    queue.push_back(Task::to_task(batch));
}

void ElementGraph::enqueue_offload_task(OffloadTask *otask, OffloadableElement *start_elem, int input_port)
{
    assert(start_elem != nullptr);
    otask->elem = start_elem;
    otask->tracker.element = (Element *) start_elem;
    otask->tracker.input_port = input_port;
    queue.push_front(Task::to_task(otask));
}

void ElementGraph::enqueue_offload_task(OffloadTask *otask, Element *start_elem, int input_port)
{
    assert(start_elem != nullptr);
    otask->elem = dynamic_cast<OffloadableElement*>(start_elem);
    otask->tracker.element = start_elem;
    otask->tracker.input_port = input_port;
    queue.push_front(Task::to_task(otask));
}

void ElementGraph::process_batch(PacketBatch *batch)
{
    Element *current_elem = batch->tracker.element;
    int input_port = batch->tracker.input_port;
    int batch_disposition = CONTINUE_TO_PROCESS;
    int64_t lb_decision = anno_get(&batch->banno, NBA_BANNO_LB_DECISION);
    uint64_t now = rdtscp();  // The starting timestamp of the current element.

    /* Check if we can and should offload. */
    if (!batch->tracker.has_results) {
        /* Since dynamic_cast has runtime overheads, we first use a bitmask
         * to check class types. */
        if (current_elem->get_type() & ELEMTYPE_OFFLOADABLE) {
            OffloadableElement *offloadable = dynamic_cast<OffloadableElement*>(current_elem);
            assert(offloadable != nullptr);
            if (lb_decision != -1) {
                /* Get or initialize the task object.
                 * This step is always executed for every input batch
                 * passing every offloadable element. */
                if (offloadable->offload(this, batch, input_port) != 0) {
                    /* We have no room for batch in the preparing task.
                     * Keep the current batch for later processing. */
                    batch->delay_start = rte_rdtsc();
                    queue.push_back(Task::to_task(batch));
                }
                /* At this point, the batch is already consumed to the task
                 * or delayed. */
                return;
            } else {
                /* If not offloaded, run the element's CPU-version handler. */
                batch_disposition = current_elem->_process_batch(input_port, batch);
                batch->compute_time += (rdtscp() - now) / batch->count;
            }
        } else {
            /* If not offloadable, run the element's CPU-version handler. */
            batch_disposition = current_elem->_process_batch(input_port, batch);
        }
    }

    /* If the element was per-batch and it said it will keep the batch,
     * we do not have to perform batch-split operations below. */
    if (batch_disposition == KEPT_BY_ELEMENT)
        return;

    /* When offloading is complete, processing of the resultant batches begins here.
     * (ref: enqueue_postproc_batch) */

    /* Here, we should have the results no matter what happened before.
     * If not, drop all packets in the batch. */
    if (!batch->tracker.has_results) {
        RTE_LOG(DEBUG, ELEM, "elemgraph: dropping a batch with no results\n");
        if (ctx->inspector) ctx->inspector->drop_pkt_count += batch->count;
        free_batch(batch);
        return;
    }

    //assert(current_elem->num_max_outputs <= num_max_outputs || current_elem->num_max_outputs == -1);
    size_t num_outputs = current_elem->next_elems.size();

    if (num_outputs == 0) {

        /* If no outputs are connected, drop all packets. */
        if (ctx->inspector) ctx->inspector->drop_pkt_count += batch->count;
        free_batch(batch);

    } else if (num_outputs == 1) {

        /* With the single output, we don't need to allocate new
         * batches.  Just reuse the given one. */
        if (0 == (current_elem->get_type() & ELEMTYPE_PER_BATCH)) {
            const int *const results = batch->results;
            #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
            batch->has_dropped = false;
            #endif
            FOR_EACH_PACKET(batch) {
                int o = results[pkt_idx];
                switch (o) {
                case 0:
                    // pass
                    break;
                #if NBA_BATCHING_SCHEME != NBA_BATCHING_CONTINUOUS
                case DROP:
                    assert(0 == rte_ring_enqueue(ctx->io_ctx->drop_queue, batch->packets[pkt_idx]));
                    EXCLUDE_PACKET(batch, pkt_idx);
                    break;
                #endif
                case PENDING:
                    // remove from PacketBatch, but don't free..
                    // They are stored in io_thread_ctx::pended_pkt_queue.
                    EXCLUDE_PACKET(batch, pkt_idx);
                    break;
                case SLOWPATH:
                    rte_panic("SLOWPATH is not supported yet. (element: %s)\n", current_elem->class_name());
                    break;
                }
            } END_FOR;
            #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
            if (batch->has_dropped)
                batch->collect_excluded_packets();
            if (ctx->inspector) ctx->inspector->drop_pkt_count += batch->drop_count;
            batch->clean_drops(ctx->io_ctx->drop_queue);
            #endif
        }
        if (unlikely(current_elem->next_elems[0]->get_type() & ELEMTYPE_OUTPUT)) {
            /* We are at the end leaf of the pipeline.
             * Inidicate free of the original batch. */
            if (ctx->inspector) {
                ctx->inspector->tx_batch_count ++;;
                ctx->inspector->tx_pkt_count += batch->count;
            }
            io_tx_batch(ctx->io_ctx, batch);
            free_batch(batch, false);
        } else {
            /* Recurse into the next element, reusing the batch. */
            Element *next_el = current_elem->next_elems[0];
            int next_input_port = current_elem->next_connected_inputs[0];

            batch->tracker.element = next_el;
            batch->tracker.input_port = next_input_port;
            batch->tracker.has_results = false;
            queue.push_back(Task::to_task(batch));
        }

    } else { /* num_outputs > 1 */

        // TODO: Work in progress!
        //size_t num_outputs = elem->next_elems.size();
        //for (unsigned o = 1; o < num_outputs; o++) {
        //}
        // TODO: zero out the used portions of elem->output_cloned_packets[]

        const int *const results = batch->results;
        PacketBatch *out_batches[num_max_outputs];
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

            memset(out_batches, 0, num_max_outputs*sizeof(PacketBatch *));
            out_batches[predicted_output] = batch;

            /* Classify packets into copy-batches. */
            #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
            batch->has_dropped = false;
            #endif
            FOR_EACH_PACKET(batch) {
                int o = results[pkt_idx];
                assert(o < (signed) num_outputs);

                /* Prediction mismatch! */
                if (unlikely(o != predicted_output)) {
                    switch(o) {
                    HANDLE_ALL_PORTS: {
                        if (!out_batches[o]) {
                            /* out_batch is not allocated yet... */
                            while (rte_mempool_get(ctx->batch_pool, (void**)(out_batches + o)) == -ENOENT
                                   && !ctx->io_ctx->loop_broken) {
                                ev_run(ctx->io_ctx->loop, 0);
                            }
                            new (out_batches[o]) PacketBatch();
                            anno_set(&out_batches[o]->banno, NBA_BANNO_LB_DECISION, lb_decision);
                            out_batches[o]->recv_timestamp = batch->recv_timestamp;
                        }
                        /* Append the packet to the output batch. */
                        ADD_PACKET(out_batches[o], batch->packets[pkt_idx]);
                        /* Exclude it from the batch. */
                        EXCLUDE_PACKET(batch, pkt_idx);
                        break; }
                    #if NBA_BATCHING_SCHEME != NBA_BATCHING_CONTINUOUS
                    case DROP:
                        rte_ring_enqueue(ctx->io_ctx->drop_queue, batch->packets[pkt_idx]);
                        EXCLUDE_PACKET(batch, pkt_idx);
                    #endif
                    case PENDING: {
                        /* The packet is stored in io_thread_ctx::pended_pkt_queue. */
                        /* Exclude it from the batch. */
                        EXCLUDE_PACKET(batch, pkt_idx);
                        break; }
                    case SLOWPATH:
                        assert(0); // Not implemented yet.
                        break;
                    }
                    current_elem->branch_miss++;
                }
                current_elem->branch_total++;
                current_elem->branch_count[o]++;
            } END_FOR;

            // NOTE: out_batches[predicted_output] == batch
            #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
            if (batch->has_dropped)
                batch->collect_excluded_packets();
            if (ctx->inspector) ctx->inspector->drop_pkt_count += batch->drop_count;
            batch->clean_drops(ctx->io_ctx->drop_queue);
            #endif

            if (current_elem->branch_total & BRANCH_TRUNC_LIMIT) {
                //double percentage = ((double)(current_elem->branch_total-current_elem->branch_miss) / (double)current_elem->branch_total);
                //printf("%s: prediction: %f\n", current_elem->class_name(), percentage);
                current_elem->branch_miss = current_elem->branch_total = 0;
                for (unsigned k = 0; k < current_elem->next_elems.size(); k++)
                    current_elem->branch_count[k] = current_elem->branch_count[k] >> 1;
            }

            /* Recurse into the element subgraph starting from each
             * output port using copy-batches. */
            for (unsigned o = 0; o < num_outputs; o++) {
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

                        out_batches[o]->tracker.element = next_el;
                        out_batches[o]->tracker.input_port = next_input_port;
                        out_batches[o]->tracker.has_results = false;

                        /* Push at the beginning of the job queue (DFS).
                         * If we insert at the end, it becomes BFS. */
                        queue.push_back(Task::to_task(out_batches[o]));
                    }
                } else {
                    /* This batch is unused! */
                    if (out_batches[o])
                        free_batch(out_batches[o]);
                }
            }
        }
#ifndef NBA_BRANCH_PREDICTION_ALWAYS
        else
#endif
#endif
#ifndef NBA_BRANCH_PREDICTION_ALWAYS
        {
            while (rte_mempool_get_bulk(ctx->batch_pool, (void **) out_batches, num_outputs) == -ENOENT
                   && !ctx->io_ctx->loop_broken) {
                ev_run(ctx->io_ctx->loop, 0);
            }

            // TODO: optimize by choosing/determining the "major" path and reuse the
            //       batch for that path.

            /* Initialize copy-batches. */
            for (unsigned o = 0; o < num_outputs; o++) {
                new (out_batches[o]) PacketBatch();
                anno_set(&out_batches[o]->banno, NBA_BANNO_LB_DECISION, lb_decision);
                out_batches[o]->recv_timestamp = batch->recv_timestamp;
            }

            /* Classify packets into copy-batches. */
            FOR_EACH_PACKET(batch) {
                int o = results[pkt_idx];
                #if NBA_BATCHING_SCHEME == NBA_BATCHING_CONTINUOUS
                if (o >= (signed) num_outputs || o < 0)
                    printf("o=%d, num_outputs=%lu, %u/%u/%u\n", o, num_outputs, pkt_idx, batch->count, batch->drop_count);
                #else
                if (o >= (signed) num_outputs || o < 0)
                    printf("o=%d, num_outputs=%lu, %u/%u\n", o, num_outputs, pkt_idx, batch->count);
                #endif
                assert(o < (signed) num_outputs && o >= 0);

                if (o >= 0) {
                    ADD_PACKET(out_batches[o], batch->packets[pkt_idx]);
                #if NBA_BATCHING_SCHEME != NBA_BATCHING_CONTINUOUS
                } else if (o == DROP) {
                    rte_ring_enqueue(ctx->io_ctx->drop_queue, batch->packets[pkt_idx]);
                #endif
                } else if (o == PENDING) {
                    // remove from PacketBatch, but don't free..
                    // They are stored in io_thread_ctx::pended_pkt_queue.
                } else if (o == SLOWPATH) {
                    assert(0);
                } else {
                    rte_panic("Invalid packet disposition value. (element: %s, value: %d)\n", current_elem->class_name(), o);
                }
                /* Packets are excluded from original batch in ALL cases. */
                /* Therefore, we do not have to collect_excluded_packets()
                 * nor clean_drops()! */
                EXCLUDE_PACKET_MARK_ONLY(batch, pkt_idx);
            } END_FOR;

            /* Recurse into the element subgraph starting from each
             * output port using copy-batches. */
            for (unsigned o = 0; o < num_outputs; o++) {
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

                        out_batches[o]->tracker.element = next_el;
                        out_batches[o]->tracker.input_port = next_input_port;
                        out_batches[o]->tracker.has_results = false;

                        /* Push at the beginning of the job queue (DFS).
                         * If we insert at the end, it becomes BFS. */
                        queue.push_back(Task::to_task(out_batches[o]));
                    }
                } else {
                    /* This batch is unused! */
                    free_batch(out_batches[o]);
                }
            }

            /* With multiple outputs (branches happened), we have made
             * copy-batches and the parent should free its batch. */
            free_batch(batch);
        }
#endif
    } /* endif(numoutputs) */
}

void ElementGraph::process_offload_task(OffloadTask *otask)
{
    uint64_t now = rte_rdtsc();
    otask->task_id += 100000; // for debugging
    otask->offload_start = now;
    send_offload_task_to_device(otask);
}

void ElementGraph::flush_tasks()
{
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
        void *raw_task = queue.front();
        queue.pop_front();
        switch (Task::get_task_type(raw_task)) {
        case TASK_SINGLE_BATCH:
          {
            PacketBatch *batch = Task::to_packet_batch(raw_task);
            process_batch(batch);
            break;
          }
        case TASK_OFFLOAD:
          {
            OffloadTask *otask = Task::to_offload_task(raw_task);
            process_offload_task(otask);
            break;
          }
        }
    } /* endwhile(queue) */
    return;
}

void ElementGraph::add_offload_action(struct offload_action_key *key)
{
    assert(offl_actions != nullptr);
    assert(rte_hash_add_key(offl_actions, key) >= 0);
}

bool ElementGraph::check_preproc(OffloadableElement *oel, int dbid)
{
#ifdef NBA_REUSE_DATABLOCKS
    struct offload_action_key key = { (void *) oel, dbid, ELEM_OFFL_PREPROC };
    return (rte_hash_lookup(offl_actions, &key) >= 0);
#else
    return true;
#endif
}

bool ElementGraph::check_postproc(OffloadableElement *oel, int dbid)
{
#ifdef NBA_REUSE_DATABLOCKS
    struct offload_action_key key = { (void *) oel, dbid, ELEM_OFFL_POSTPROC };
    return (rte_hash_lookup(offl_actions, &key) >= 0);
#else
    return true;
#endif
}

bool ElementGraph::check_postproc_all(OffloadableElement *oel)
{
#ifdef NBA_REUSE_DATABLOCKS
    struct offload_action_key key = { (void *) oel, -1, ELEM_OFFL_POSTPROC_FIN };
    return (rte_hash_lookup(offl_actions, &key) >= 0);
#else
    return true;
#endif
}

bool ElementGraph::check_next_offloadable(Element *offloaded_elem)
{
    return ((offloaded_elem->next_elems[0]->get_type() & ELEMTYPE_OFFLOADABLE) != 0);
}

Element *ElementGraph::get_first_next(Element *elem)
{
    return elem->next_elems[0];
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
    if (new_elem->get_type() & ELEMTYPE_OFFLOADABLE) {
        auto oelem = dynamic_cast<OffloadableElement*> (new_elem);
        assert(oelem != nullptr);
        offl_elements.push_back(oelem);
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
        for (auto el : elements) {
            if (from_elem == el) {
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

int ElementGraph::validate()
{
    // TODO: implement
    return 0;
}

const FixedRing<Element*>& ElementGraph::get_elements() const
{
    return elements;
}

// vim: ts=8 sts=4 sw=4 et
