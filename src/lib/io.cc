/**
 * NBA's RX/TX common components and IO loop
 *
 * Author: Joongi Kim <joongi@an.kaist.ac.kr>
 */

#include <nba/core/intrinsic.hh>
#include <nba/core/threading.hh>
#include <nba/core/timing.hh>
#include <nba/core/logging.hh>
#include <nba/framework/config.hh>
#include <nba/framework/logging.hh>
#include <nba/framework/io.hh>
#include <nba/framework/threadcontext.hh>
#include <nba/framework/datablock.hh>
#include <nba/framework/task.hh>
#include <nba/framework/offloadtask.hh>
#include <nba/framework/loadbalancer.hh>
#include <nba/framework/elementgraph.hh>
#include <nba/framework/computecontext.hh>
#include <nba/element/packet.hh>
#include <nba/element/packetbatch.hh>

#ifdef NBA_CPU_MICROBENCH
#include <papi.h>
#endif
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <syscall.h>
#include <numa.h>
#include <sys/prctl.h>
#include <rte_config.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_atomic.h>
#include <rte_byteorder.h>
#include <rte_errno.h>
#include <rte_memory.h>
#include <rte_memzone.h>
#include <rte_malloc.h>
#include <rte_memcpy.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_byteorder.h>
#include <rte_ethdev.h>
#include <rte_prefetch.h>
#include <rte_cycles.h>
#include <rte_ring.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <ev.h>
#ifdef USE_NVPROF
#include <nvToolsExt.h>
#endif

#include <functional>
#include <random>

using namespace std;
using namespace nba;

namespace nba {

static thread_local uint64_t recv_batch_cnt = 0;

#ifdef TEST_MINIMAL_L2FWD
struct packet_batch {
    unsigned count;
    struct rte_mbuf *pkts[NBA_MAX_COMP_BATCH_SIZE];
};
#endif

typedef function<uint32_t(void)> random32_func_t;
typedef function<uint64_t(void)> random64_func_t;
typedef function<void(char*, int, int, random32_func_t)> packet_builder_func_t;

/* ===== COMP ===== */
static void comp_packetbatch_init(struct rte_mempool *mp, void *arg, void *obj, unsigned idx)
{
    PacketBatch *b = (PacketBatch *) obj;
    new (b) PacketBatch();
}

static void comp_dbstate_init(struct rte_mempool *mp, void *arg, void *obj, unsigned idx)
{
    memset(obj, 0, sizeof(struct datablock_tracker) * NBA_MAX_DATABLOCKS);
}

static void comp_task_init(struct rte_mempool *mp, void *arg, void *obj, unsigned idx)
{
    OffloadTask *t = (OffloadTask *) obj;
    new (t) OffloadTask();
}

static void comp_prepare_cb(struct ev_loop *loop, struct ev_check *watcher, int revents)
{
    /* This routine is called when ev_run() is about to block.
     * (i.e., there is no outstanding events)
     * Calling this there allows to check any pending tasks so that
     * we could eventually release any resources such as batch objects
     * and allow other routines waiting for their releases to continue. */
    io_thread_context *io_ctx = (io_thread_context *) ev_userdata(loop);
    comp_thread_context *ctx = io_ctx->comp_ctx;
    ctx->elem_graph->flush_tasks();
    ctx->elem_graph->scan_offloadable_elements(0);
}

static void comp_offload_task_completion_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)
{
    #ifdef USE_NVPROF
    nvtxRangePush("task_completion_cb");
    #endif
    io_thread_context *io_ctx = (io_thread_context *) ev_userdata(loop);
    comp_thread_context *ctx = io_ctx->comp_ctx;
    OffloadTask *tasks[ctx->task_completion_queue_size];
    unsigned nr_tasks = rte_ring_sc_dequeue_burst(ctx->task_completion_queue,
                                                  (void **) tasks,
                                                  ctx->task_completion_queue_size);
    print_ratelimit("# done tasks", nr_tasks, 100);

    for (unsigned t = 0; t < nr_tasks && !io_ctx->loop_broken; t++) {
        /* We already finished postprocessing.
         * Retrieve the task and results. */
        uint64_t now = rdtscp();
        OffloadTask *task = tasks[t];
        ComputeContext *cctx = task->cctx;
        #ifdef USE_NVPROF
        nvtxRangePush("task");
        #endif

        /* Run postprocessing handlers. */
        task->postprocess();

        if (ctx->elem_graph->check_postproc_all(task->elem)) {
            /* Reset all datablock trackers. */
            for (PacketBatch *batch : task->batches) {
                if (batch->datablock_states != nullptr) {
                    struct datablock_tracker *t = batch->datablock_states;
                    rte_mempool_put(ctx->dbstate_pool, (void *) t);
                    batch->datablock_states = nullptr;
                }
            }
            /* Release per-task io_base. */
            task->cctx->clear_io_buffers(task->io_base);
            ev_break(ctx->io_ctx->loop, EVBREAK_ALL);
        }

        /* Update statistics. */
        uint64_t task_cycles = now - task->offload_start;
        float time_spent = (float) task_cycles / rte_get_tsc_hz();
        uint64_t task_count = ctx->inspector->dev_finished_task_count[task->local_dev_idx];
        ctx->inspector->avg_task_completion_sec[task->local_dev_idx] \
              = (ctx->inspector->avg_task_completion_sec[task->local_dev_idx] * task_count + time_spent) / (task_count + 1);
        ctx->inspector->dev_finished_task_count[task->local_dev_idx] ++;
        ctx->inspector->dev_finished_batch_count[task->local_dev_idx] += task->batches.size();

        /* Enqueue batches for later processing. */
        uint64_t total_batch_size = 0;
        for (PacketBatch *batch : task->batches)
            total_batch_size += batch->count;
        #ifdef NBA_REUSE_DATABLOCKS
        if (ctx->elem_graph->check_next_offloadable(task->elem)) {
            for (PacketBatch *batch : task->batches) {
                batch->compute_time += (uint64_t)
                        ((float) task_cycles / total_batch_size
                         - ((float) batch->delay_time / batch->count));
            }
            /* Rewind the state so that it gets "prepared" by ElemGraph.
             * (e.g., update datablock list used by next element) */
            task->state = TASK_INITIALIZED;
            ctx->elem_graph->enqueue_offload_task(task,
                                                  ctx->elem_graph->get_first_next(task->elem),
                                                  0);
            /* This task is reused. We keep them intact. */
        } else {
        #else
        {
        #endif
            for (size_t b = 0, b_max = task->batches.size(); b < b_max; b ++) {
                task->batches[b]->compute_time += (uint64_t)
                        ((float) task_cycles / total_batch_size
                         - ((float) task->batches[b]->delay_time / task->batches[b]->count));
                task->elem->enqueue_batch(task->batches[b]);
            }

            /* Free the task object. */
            task->cctx = nullptr;
            task->~OffloadTask();
            rte_mempool_put(ctx->task_pool, (void *) task);
        }

        /* Free the resources used for this offload task. */
        cctx->currently_running_task = nullptr;
        cctx->state = ComputeContext::READY;

        #ifdef USE_NVPROF
        nvtxRangePop();
        #endif
    }
    #ifdef USE_NVPROF
    nvtxRangePop();
    #endif
}

static size_t comp_process_batch(io_thread_context *ctx, void *pkts, size_t count, uint64_t loop_count)
{
    assert(count <= ctx->comp_ctx->num_combatch_size);
    if (count == 0) return 0;
    int ret;
    PacketBatch *batch = nullptr;
    while (true) {
        ret = rte_mempool_get(ctx->comp_ctx->batch_pool, (void **) &batch);
        if (unlikely(ctx->loop_broken)) return 0;
        if (ret == -ENOENT) {
            /* Wait until some batches are freed. */
            ev_run(ctx->loop, 0);
        } else
            break;
    }

    /* Okay, let's initialize a new packet batch. */
    assert(batch != nullptr);
    new (batch) PacketBatch();
    memcpy((void **) &batch->packets[0], (void **) pkts, count * sizeof(void*));
    batch->banno.bitmask = 0;
    anno_set(&batch->banno, NBA_BANNO_LB_DECISION, -1);

    /* t is NOT the actual receive timestamp but a
     * "start-of-processing" timestamp.
     * However its ordering is same as we do FIFO here.
     */
    uint64_t t = rdtscp();
    batch->count = count;
    INIT_BATCH_MASK(batch);
    batch->recv_timestamp = t;
    batch->compute_time = 0;
    batch->delay_start = 0;
    batch->delay_time = 0;
    batch->batch_id = recv_batch_cnt;
    #if NBA_BATCHING_SCHEME == NBA_BATCHING_LINKEDLIST
    batch->first_idx = 0;
    batch->last_idx = batch->count - 1;
    batch->slot_count = batch->count;
    Packet *prev_pkt = nullptr;
    #endif
    FOR_EACH_PACKET_ALL_INIT_PREFETCH(batch, 8u) {
        /* Initialize packet metadata objects in pktmbuf's private area. */
        Packet *pkt = Packet::from_base_nocheck(batch->packets[pkt_idx]);
        new (pkt) Packet(batch, batch->packets[pkt_idx]);
        #if NBA_BATCHING_SCHEME == NBA_BATCHING_LINKEDLIST
        if (prev_pkt != nullptr) {
            prev_pkt->next_idx = pkt_idx;
            pkt->prev_idx = pkt_idx - 1;
        }
        prev_pkt = pkt;
        #endif

        /* Set annotations and strip the temporary headroom. */
        pkt->anno.bitmask = 0;
        anno_set(&pkt->anno, NBA_ANNO_IFACE_IN,
                 batch->packets[pkt_idx]->port);
        anno_set(&pkt->anno, NBA_ANNO_TIMESTAMP, t);
        anno_set(&pkt->anno, NBA_ANNO_BATCH_ID, recv_batch_cnt);
    } END_FOR_ALL_INIT_PREFETCH;
    recv_batch_cnt ++;

    /* Run the element graph's schedulable elements.
     * FIXME: allow multiple FromInput elements depending on the flow groups. */
    ctx->comp_ctx->elem_graph->feed_input(0, batch, loop_count);
    return count;
}
/* ===== END_OF_COMP ===== */

/* Taken from PSIO */
static inline uint16_t ip_fast_csum(const void *iph, unsigned int ihl)/*{{{*/
{
    unsigned int sum;

    asm("  movl (%1), %0\n"
        "  subl $4, %2\n"
        "  jbe 2f\n"
        "  addl 4(%1), %0\n"
        "  adcl 8(%1), %0\n"
        "  adcl 12(%1), %0\n"
        "1: adcl 16(%1), %0\n"
        "  lea 4(%1), %1\n"
        "  decl %2\n"
        "  jne      1b\n"
        "  adcl $0, %0\n"
        "  movl %0, %2\n"
        "  shrl $16, %0\n"
        "  addw %w2, %w0\n"
        "  adcl $0, %0\n"
        "  notl %0\n"
        "2:"
        /* Since the input registers which are loaded with iph and ih
           are modified, we must also specify them as outputs, or gcc
           will assume they contain their original values. */
        : "=r" (sum), "=r" (iph), "=r" (ihl)
        : "1" (iph), "2" (ihl)
           : "memory");
    return (uint16_t) sum;
}/*}}}*/
static inline uint32_t io_myrand(uint64_t *seed) /*{{{*/
{
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t)(*seed >> 32);
}/*}}}*/

static void io_local_stat_timer_cb(struct ev_loop *loop, struct ev_timer *watcher, int revents)/*{{{*/
{
    io_thread_context *ctx = (io_thread_context *) ev_userdata(loop);
    ctx->tx_pkt_thruput = 0;
    /* Atomically update the counters in the master. */
    for (unsigned j = 0; j < ctx->node_stat->num_ports; j++) {
        rte_atomic64_add(&ctx->node_stat->port_stats[j].num_recv_pkts, ctx->port_stats[j].num_recv_pkts);
        rte_atomic64_add(&ctx->node_stat->port_stats[j].num_sent_pkts, ctx->port_stats[j].num_sent_pkts);
        rte_atomic64_add(&ctx->node_stat->port_stats[j].num_sw_drop_pkts, ctx->port_stats[j].num_sw_drop_pkts);
        rte_atomic64_add(&ctx->node_stat->port_stats[j].num_rx_drop_pkts, ctx->port_stats[j].num_rx_drop_pkts);
        rte_atomic64_add(&ctx->node_stat->port_stats[j].num_tx_drop_pkts, ctx->port_stats[j].num_tx_drop_pkts);
        rte_atomic64_add(&ctx->node_stat->port_stats[j].num_invalid_pkts, ctx->port_stats[j].num_invalid_pkts);
        rte_atomic64_add(&ctx->node_stat->port_stats[j].num_recv_bytes, ctx->port_stats[j].num_recv_bytes);
        rte_atomic64_add(&ctx->node_stat->port_stats[j].num_sent_bytes, ctx->port_stats[j].num_sent_bytes);
        ctx->tx_pkt_thruput += ctx->port_stats[j].num_sent_pkts;
        memset(&ctx->port_stats[j], 0, sizeof(struct io_port_stat));
    }
 #ifdef NBA_CPU_MICROBENCH
	char buf[2048];
    char *bufp = &buf[0];
    for (int e = 0; e < 5; e++) {
        bufp += sprintf(bufp, "[worker:%02u].%d %'12lld, %'12lld, %'12lld\n", ctx->loc.core_id, e, ctx->papi_ctr_rx[e], ctx->papi_ctr_tx[e], ctx->papi_ctr_comp[e]);
    }
	printf("%s", buf);
    memset(ctx->papi_ctr_rx, 0, sizeof(long long) * 5);
    memset(ctx->papi_ctr_tx, 0, sizeof(long long) * 5);
    memset(ctx->papi_ctr_comp, 0, sizeof(long long) * 5);
#endif
    /* Inform the master to check updates. */
    rte_atomic16_inc(ctx->node_master_flag);
    ev_async_send(ctx->node_master_ctx->loop, ctx->node_stat_watcher);
    /* Re-arm the timer. */
    ev_timer_again(loop, watcher);
}/*}}}*/

static void io_node_stat_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)/*{{{*/
{
    io_thread_context *ctx = (io_thread_context *) ev_userdata(loop);
    /* node_stat is a shared structure. */
    struct io_node_stat *node_stat = (struct io_node_stat *) ctx->node_stat;

    /* All threads must have reported the stats. */
    if (rte_atomic16_cmpset((volatile uint16_t *) &ctx->node_master_flag->cnt, node_stat->num_threads, 0)) {
        unsigned j;
        struct io_thread_stat total;
        struct io_thread_stat *last_total = &node_stat->last_total;
        struct rte_eth_stats s;
        for (j = 0; j < node_stat->num_ports; j++) {
            struct rte_eth_dev_info info;
            rte_eth_dev_info_get((uint8_t) j, &info);
            total.port_stats[j].num_recv_pkts = rte_atomic64_read(&node_stat->port_stats[j].num_recv_pkts);
            total.port_stats[j].num_sent_pkts = rte_atomic64_read(&node_stat->port_stats[j].num_sent_pkts);
            total.port_stats[j].num_recv_bytes = rte_atomic64_read(&node_stat->port_stats[j].num_recv_bytes);
            total.port_stats[j].num_sent_bytes = rte_atomic64_read(&node_stat->port_stats[j].num_sent_bytes);
            total.port_stats[j].num_invalid_pkts = rte_atomic64_read(&node_stat->port_stats[j].num_invalid_pkts);
            total.port_stats[j].num_sw_drop_pkts = rte_atomic64_read(&node_stat->port_stats[j].num_sw_drop_pkts);
            if ((unsigned) rte_eth_dev_socket_id(j) == ctx->loc.node_id) {
                rte_eth_stats_get((uint8_t) j, &s);
                total.port_stats[j].num_rx_drop_pkts = s.ierrors;
            }
            total.port_stats[j].num_tx_drop_pkts = rte_atomic64_read(&node_stat->port_stats[j].num_tx_drop_pkts);
        }
        uint64_t cur_time = get_usec();
        double total_thruput_mpps = 0;
        double total_thruput_gbps = 0;
        double port_thruput_mpps, port_thruput_gbps;
        for (j = 0; j < node_stat->num_ports; j++) {
            port_thruput_mpps = 0;
            port_thruput_gbps = 0;
            printf("port[%u:%u]: %'10lu %'10lu %'10lu %'10lu %'10lu %'10lu %'10lu %'10lu | forwarded %2.2f Mpps, %2.2f Gbps \n",
                   node_stat->node_id, j,
                   total.port_stats[j].num_recv_pkts - last_total->port_stats[j].num_recv_pkts,
                   (total.port_stats[j].num_recv_bytes - last_total->port_stats[j].num_recv_bytes) << 3,
                   total.port_stats[j].num_sent_pkts - last_total->port_stats[j].num_sent_pkts,
                   (total.port_stats[j].num_sent_bytes - last_total->port_stats[j].num_sent_bytes) << 3,
                   total.port_stats[j].num_invalid_pkts - last_total->port_stats[j].num_invalid_pkts,
                   total.port_stats[j].num_sw_drop_pkts - last_total->port_stats[j].num_sw_drop_pkts,
                   total.port_stats[j].num_rx_drop_pkts - last_total->port_stats[j].num_rx_drop_pkts,
                   total.port_stats[j].num_tx_drop_pkts - last_total->port_stats[j].num_tx_drop_pkts,
                   (port_thruput_mpps = ((double)total.port_stats[j].num_sent_pkts - last_total->port_stats[j].num_sent_pkts) / (cur_time - node_stat->last_time)),
                   (port_thruput_gbps = ((double)((total.port_stats[j].num_sent_bytes - last_total->port_stats[j].num_sent_bytes) << 3)) / ((cur_time - node_stat->last_time)*1000)));
            total_thruput_mpps += port_thruput_mpps;
            total_thruput_gbps += port_thruput_gbps;
        }
        printf("Total forwarded pkts: %.2f Mpps, %.2f Gbps in node %d\n", total_thruput_mpps, total_thruput_gbps, node_stat->node_id);
        rte_memcpy(last_total, &total, sizeof(total));
        node_stat->last_time = get_usec();
        fflush(stdout);
    }
}/*}}}*/

static void io_terminate_cb(struct ev_loop *loop, struct ev_async *watcher, int revents)
{
    struct io_thread_context *ctx = (struct io_thread_context *) ev_userdata(loop);
    ctx->loop_broken = true;
    ev_break(loop, EVBREAK_ALL);
}

/**
 * The TXCommonComponent implementation.
 * This function is directly called from the computation thread.
 */
void io_tx_batch(struct io_thread_context *ctx, PacketBatch *batch)
{
    struct rte_mbuf *out_batches[NBA_MAX_PORTS][NBA_MAX_COMP_BATCH_SIZE];
    unsigned out_batches_cnt[NBA_MAX_PORTS];
    memset(out_batches_cnt, 0, sizeof(unsigned) * NBA_MAX_PORTS);
    uint64_t t = rdtscp();
    int64_t proc_id = anno_get(&batch->banno, NBA_BANNO_LB_DECISION) + 1; // adjust range to be positive
    ctx->comp_ctx->inspector->update_batch_proc_time(t - batch->recv_timestamp);
    ctx->comp_ctx->inspector->update_pkt_proc_cycles(batch->compute_time, proc_id);
//#ifdef NBA_CPU_MICROBENCH
//    PAPI_start(ctx->papi_evset_tx);
//#endif

    // TODO: keep ordering of packets (or batches)
    //   NOTE: current implementation: no extra queueing,
    //   just transmit as requested
    FOR_EACH_PACKET(batch) {
        Packet *pkt = Packet::from_base(batch->packets[pkt_idx]);
        struct ether_hdr *ethh = rte_pktmbuf_mtod(batch->packets[pkt_idx], struct ether_hdr *);
        uint64_t o = anno_get(&pkt->anno, NBA_ANNO_IFACE_OUT);

        /* Update source/dest MAC addresses. */
        ether_addr_copy(&ethh->s_addr, &ethh->d_addr);
        ether_addr_copy(&ctx->tx_ports[o].addr, &ethh->s_addr);

        /* Append to the corresponding output batch. */
        int cnt = out_batches_cnt[o] ++;
        out_batches[o][cnt] = batch->packets[pkt_idx];
    } END_FOR;

    unsigned tx_tries = 0;
    for (unsigned o = 0; o < ctx->num_tx_ports; o++) {
        if (out_batches_cnt[o] == 0)
            continue;
        struct rte_mbuf **pkts = out_batches[o];
        unsigned count = out_batches_cnt[o];
        uint64_t total_sent_byte = 0;

        /* Sum TX packet bytes. */
        for(unsigned k = 0; k < count; k++) {
            struct rte_mbuf* cur_pkt = out_batches[o][k];
            unsigned len = rte_pktmbuf_pkt_len(cur_pkt) + 24;  /* Add Ethernet overheads */
            ctx->port_stats[o].num_sent_bytes += len;
        }

#if NBA_OQ
        /* To implement output-queuing, we need to drop when the TX NIC
         * is congested.  This would not happen in high line rates such
         * as 10 GbE because processing speed becomes the bottleneck,
         * but it will be meaningful when we use low-speed NICs such as
         * 1 GbE cards. */
        unsigned txq = ctx->loc.global_thread_idx;
        unsigned sent_cnt = rte_eth_tx_burst((uint8_t) o, txq, pkts, count);
        for (unsigned k = sent_cnt; k < count; k++) {
            struct rte_mbuf* cur_pkt = out_batches[o][k];
            unsigned len = rte_pktmbuf_pkt_len(cur_pkt) + 24;
            ctx->port_stats[o].num_sent_bytes -= len;
            rte_pktmbuf_free(pkts[k]);
        }
        ctx->port_stats[o].num_sent_pkts += sent_cnt;
        ctx->port_stats[o].num_tx_drop_pkts += (count - sent_cnt);
        ctx->global_tx_cnt += sent_cnt;
#else
        /* Try to send all packets with retries. */
        unsigned total_sent_cnt = 0;
        do {
            unsigned txq = ctx->loc.global_thread_idx;
            unsigned sent_cnt = rte_eth_tx_burst((uint8_t) o, txq, &pkts[total_sent_cnt], count);
            count -= sent_cnt;
            total_sent_cnt += sent_cnt;
            tx_tries ++;
        } while (count > 0);
        ctx->port_stats[o].num_sent_pkts += total_sent_cnt;
        ctx->global_tx_cnt += total_sent_cnt;
#endif
    }
//#ifdef NBA_CPU_MICROBENCH
//    {
//        long long ctr[5];
//        PAPI_stop(ctx->papi_evset_tx, ctr);
//        for (int i = 0; i < 5; i++)
//            ctx->papi_ctr_tx[i] += ctr[i];
//    }
//#endif
    print_ratelimit("# tx trials per batch", tx_tries, 10000);
}

int io_loop(void *arg)
{
    struct io_thread_context *const ctx = (struct io_thread_context *) arg;

    /* Ensure we are on the right core. */
    assert(rte_socket_id() == ctx->loc.node_id);
    assert(rte_lcore_id() == ctx->loc.core_id);
    RTE_LOG(DEBUG, IO, "@%u: starting to read from %u hw rx queue(s)\n",
               ctx->loc.core_id, ctx->num_hw_rx_queues);
    ctx->thread_id = syscall(SYS_gettid);

    // the way numa index numbered for each cpu core is checked in main(). (see 'is_numa_idx_grouped' in main())
    const unsigned num_nodes = numa_num_configured_nodes();
    struct rte_mbuf *pkts[NBA_MAX_IO_BATCH_SIZE * NBA_MAX_QUEUES_PER_PORT];
    struct rte_mbuf *drop_pkts[NBA_MAX_IO_BATCH_SIZE];
    struct timespec sleep_ts;
    unsigned i, j;
    char temp[1024];
    int ret;

    /* Initialize random for randomized port access order. */
    random32_func_t random32 = bind(uniform_int_distribution<uint32_t>{}, mt19937());

    /* IO thread initialization */
    assert((unsigned) numa_node_of_cpu(ctx->loc.core_id) == ctx->loc.node_id);

    snprintf(temp, 64, "compio.%u:%u@%u", ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
    prctl(PR_SET_NAME, temp, 0, 0, 0);
    threading::bind_cpu(ctx->loc.core_id);
    #ifdef USE_NVPROF
    nvtxNameOsThread(pthread_self(), temp);
    #endif

#ifdef NBA_CPU_MICROBENCH
    assert(PAPI_register_thread() == PAPI_OK);
    ctx->papi_evset_rx = PAPI_NULL;
    ctx->papi_evset_tx = PAPI_NULL;
    ctx->papi_evset_comp = PAPI_NULL;
    memset(ctx->papi_ctr_rx, 0, sizeof(long long) * 5);
    memset(ctx->papi_ctr_tx, 0, sizeof(long long) * 5);
    memset(ctx->papi_ctr_comp, 0, sizeof(long long) * 5);
    assert(PAPI_create_eventset(&ctx->papi_evset_rx) == PAPI_OK);
    assert(PAPI_create_eventset(&ctx->papi_evset_tx) == PAPI_OK);
    assert(PAPI_create_eventset(&ctx->papi_evset_comp) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_rx, PAPI_TOT_CYC) == PAPI_OK);   // total cycles
    assert(PAPI_add_event(ctx->papi_evset_rx, PAPI_TOT_INS) == PAPI_OK);   // total instructions
    assert(PAPI_add_event(ctx->papi_evset_rx, PAPI_L2_TCM) == PAPI_OK);   // total load/store instructions
    assert(PAPI_add_event(ctx->papi_evset_rx, PAPI_BR_CN) == PAPI_OK);    // number of conditional branches
    assert(PAPI_add_event(ctx->papi_evset_rx, PAPI_BR_MSP) == PAPI_OK);    // number of branch misprediction
    assert(PAPI_add_event(ctx->papi_evset_tx, PAPI_TOT_CYC) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_tx, PAPI_TOT_INS) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_tx, PAPI_L2_TCM) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_tx, PAPI_BR_CN) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_tx, PAPI_BR_MSP) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_comp, PAPI_TOT_CYC) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_comp, PAPI_TOT_INS) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_comp, PAPI_L2_TCM) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_comp, PAPI_BR_CN) == PAPI_OK);
    assert(PAPI_add_event(ctx->papi_evset_comp, PAPI_BR_MSP) == PAPI_OK);
#endif

    /* Read TX-port MAC addresses. */
    for (i = 0; i < ctx->num_tx_ports; i++) {
        ctx->tx_ports[i].port_idx = i;
        rte_eth_macaddr_get(i, &ctx->tx_ports[i].addr);
    }

    /* Initialize the event loop. */
    ctx->loop = ev_loop_new(EVFLAG_AUTO | EVFLAG_NOSIGMASK);
    ctx->loop_broken = false;
    ev_set_userdata(ctx->loop, ctx);

    /* ==== COMP ====*/
    ctx->comp_ctx->loop = ctx->loop;
    snprintf(temp, RTE_MEMPOOL_NAMESIZE,
         "comp.batch.%u:%u@%u", ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
    ctx->comp_ctx->batch_pool = rte_mempool_create(temp, ctx->comp_ctx->num_batchpool_size + 1,
                                                   sizeof(PacketBatch), CACHE_LINE_SIZE,
                                                   0, nullptr, nullptr,
                                                   comp_packetbatch_init, nullptr,
                                                   ctx->loc.node_id, 0);
    if (ctx->comp_ctx->batch_pool == nullptr)
        rte_panic("RTE_ERROR while creating comp_ctx->batch_pool: %s\n", rte_strerror(rte_errno));

    snprintf(temp, RTE_MEMPOOL_NAMESIZE,
        "comp.dbstate.%u:%u@%u", ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
    size_t dbstate_pool_size = NBA_MAX_COPROC_PPDEPTH * 16;
    size_t dbstate_item_size = sizeof(struct datablock_tracker) * NBA_MAX_DATABLOCKS;
    ctx->comp_ctx->dbstate_pool = rte_mempool_create(temp, dbstate_pool_size + 1,
                                                     dbstate_item_size, 32,
                                                     0, nullptr, nullptr,
                                                     comp_dbstate_init, nullptr,
                                                     ctx->loc.node_id, 0);
    if (ctx->comp_ctx->dbstate_pool == nullptr) {
        //printf("sizeof(struct datablock_tracker) = %'lu\n", sizeof(struct datablock_tracker));
        rte_panic("RTE_ERROR while creating comp_ctx->dbstate_pool: %s\n", rte_strerror(rte_errno));
    }

    snprintf(temp, RTE_MEMPOOL_NAMESIZE,
         "comp.task.%u:%u@%u", ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
    ctx->comp_ctx->task_pool = rte_mempool_create(temp, ctx->comp_ctx->num_taskpool_size + 1,
                                                  sizeof(OffloadTask), 32,
                                                  0, nullptr, nullptr,
                                                  comp_task_init, nullptr,
                                                  ctx->loc.node_id, 0);
    if (ctx->comp_ctx->task_pool == nullptr)
        rte_panic("RTE_ERROR while creating comp_ctx->task pool: %s\n", rte_strerror(rte_errno));

    ctx->comp_ctx->packet_pool = packet_create_mempool(128, ctx->loc.node_id, ctx->loc.core_id);
    assert(ctx->comp_ctx->packet_pool != nullptr);

    NEW(ctx->loc.node_id, ctx->comp_ctx->inspector, SystemInspector);

    /* Register the offload completion event. */
    if (ctx->comp_ctx->coproc_ctx != nullptr) {
        ev_async_init(ctx->comp_ctx->task_completion_watcher, comp_offload_task_completion_cb);
        // TODO: remove this event and just check the completion queue on every iteration.
        ev_async_start(ctx->loop, ctx->comp_ctx->task_completion_watcher);
    }

    /* Register per-iteration check event. */
    ctx->comp_ctx->check_watcher = (struct ev_check *) rte_malloc_socket(nullptr, sizeof(struct ev_check),
                                                                         CACHE_LINE_SIZE, ctx->loc.node_id);
    ev_check_init(ctx->comp_ctx->check_watcher, comp_prepare_cb);
    ev_check_start(ctx->loop, ctx->comp_ctx->check_watcher);

    /* ==== END_OF_COMP ====*/

    /* Register the termination event. */
    ev_set_cb(ctx->terminate_watcher, io_terminate_cb);
    ev_async_start(ctx->loop, ctx->terminate_watcher);

    /* Initialize statistics. */
    ctx->port_stats = (struct io_port_stat *) rte_malloc_socket("io_port_stat",
                                                                sizeof(struct io_port_stat) * ctx->node_stat->num_ports,
                                                                CACHE_LINE_SIZE, ctx->loc.node_id);
    memset(ctx->port_stats, 0, sizeof(struct io_port_stat) * ctx->node_stat->num_ports);

    /* Initialize statistics timer. */
    if (ctx->loc.local_thread_idx == 0) {

        *ctx->node_master_flag = RTE_ATOMIC16_INIT(0);
        ev_async_init(ctx->node_stat_watcher, io_node_stat_cb);
        ev_async_start(ctx->loop, ctx->node_stat_watcher);

        ctx->init_cond->lock();
        *ctx->init_done = true;
        ctx->init_cond->signal_all();
        ctx->init_cond->unlock();
    } else {
        ctx->init_cond->lock();
        while (!*(ctx->init_done)) {
            ctx->init_cond->wait();
        }
        ctx->init_cond->unlock();
    }
    ctx->stat_timer = (struct ev_timer *) rte_malloc_socket("io_stat_timer", sizeof(struct ev_timer),
                                                            CACHE_LINE_SIZE, ctx->loc.node_id);
    ev_init(ctx->stat_timer, io_local_stat_timer_cb);
    ctx->stat_timer->repeat = 1.;
    ev_timer_again(ctx->loop, ctx->stat_timer);

#ifdef TEST_MINIMAL_L2FWD
    unsigned txq = (ctx->loc.node_id * (ctx->num_tx_ports / num_nodes)) + ctx->loc.local_thread_idx;
    struct packet_batch *batch = (struct packet_batch *) rte_malloc_socket("pktbatch",
            sizeof(*batch), 64, ctx->loc.node_id);
    batch->count= 0;
    unsigned next_ports[ctx->num_tx_ports];
    for (i = 0; i < ctx->num_tx_ports; i++)
        next_ports[i] = 0;
#endif
#ifdef NBA_RANDOM_PORT_ACCESS
    uint32_t magic = 0x7ED996DB;
    int random_mapping[NBA_MAX_PORTS];
    for (i=0; i < NBA_MAX_PORTS; i++)
        random_mapping[i] = i;
#endif

    int num_random_packets = 4096;
    int rp = 0;
    char *random_packets[4096];

    // ctx->num_iobatch_size = 1; // FOR TESTING
    uint64_t loop_count = 0;

    /* The IO thread runs in polling mode. */
    while (likely(!ctx->loop_broken)) {
        unsigned total_recv_cnt = 0;
        #ifdef NBA_CPU_MICROBENCH/*{{{*/
        PAPI_start(ctx->papi_evset_rx);
        #endif/*}}}*/
        for (i = 0; i < ctx->num_hw_rx_queues; i++) {
#ifdef NBA_RANDOM_PORT_ACCESS /*{{{*/
            /* Shuffle the RX queue list. */
            int swap_idx = random32() % ctx->num_hw_rx_queues;
            int temp = random_mapping[i];
            random_mapping[i] = random_mapping[swap_idx];
            random_mapping[swap_idx] = temp;
        }
        unsigned _temp;
        for (_temp = 0; _temp < ctx->num_hw_rx_queues; _temp++) {
            i = random_mapping[_temp];
#endif /*}}}*/
            unsigned port_idx = ctx->rx_hwrings[i].ifindex;
            unsigned rxq      = ctx->rx_hwrings[i].qidx;
            unsigned recv_cnt = 0;
            unsigned sent_cnt = 0, invalid_cnt = 0;

            recv_cnt = rte_eth_rx_burst((uint8_t) port_idx, rxq,
                                         &pkts[total_recv_cnt], ctx->num_iobatch_size);

#if !defined(TEST_RXONLY) && !defined(TEST_MINIMAL_L2FWD)
            for(unsigned _k=0; _k<recv_cnt; _k++)
            {
                struct rte_mbuf* cur_pkt = pkts[total_recv_cnt + _k];
                ctx->port_stats[port_idx].num_recv_bytes += rte_pktmbuf_pkt_len(cur_pkt) + 24;
            }
            total_recv_cnt += recv_cnt;
            ctx->port_stats[port_idx].num_recv_pkts += recv_cnt;
#endif
#ifdef TEST_RXONLY/*{{{*/
            /* Drop all packets in software */
            for (j = 0; j < recv_cnt; j++)
                rte_pktmbuf_free(pkts[j]);
            ctx->port_stats[port_idx].num_sw_drop_pkts += recv_cnt;
#endif/*}}}*/
#ifdef TEST_MINIMAL_L2FWD/*{{{*/
            /* Minimal L2 forwarding */
            for (j = 0; j < recv_cnt; j++) {
                struct ether_hdr *ethh;
                char temp[ETHER_ADDR_LEN];
                rte_prefetch0(rte_pktmbuf_mtod(pkts[j], void *));
                if (j < recv_cnt - 1)
                    rte_prefetch0(rte_pktmbuf_mtod(pkts[j + 1], void *));
                ethh = rte_pktmbuf_mtod(pkts[j], struct ether_hdr *);

                if (likely(is_unicast_ether_addr(&ethh->d_addr))) {
                    /* Update source/dest MAC addresses to echo back. */
                    ether_addr_copy(&ethh->s_addr, &ethh->d_addr);
                    ether_addr_copy(&ctx->tx_ports[port_idx].addr, &ethh->s_addr);
                    batch->pkts[batch->count ++] = pkts[j];
                } else {
                    /* Skip non-unicast packets. */
                    rte_pktmbuf_free(pkts[j]);
                    invalid_cnt ++;
                }
            }

            unsigned total_sent_cnt = 0;
            // NOTE: To round-robin over all ports including TX NUMA node crossing:
            // next_ports[port_idx] = (next_ports[port_idx] + 1) % ctx->num_ports;
            next_ports[port_idx] = (next_ports[port_idx] + 1) % (ctx->num_tx_ports / num_nodes)
                                   + ctx->loc.node_id * (ctx->num_tx_ports / num_nodes);
            unsigned out_port_idx = next_ports[port_idx];
            do {
                sent_cnt = rte_eth_tx_burst((uint8_t) out_port_idx, txq,
                                            batch->pkts + total_sent_cnt, batch->count);
                batch->count -= sent_cnt;
                total_sent_cnt += sent_cnt;
            } while (batch->count > 0);
            ctx->port_stats[port_idx].num_recv_pkts += recv_cnt;
            ctx->port_stats[port_idx].num_invalid_pkts += invalid_cnt;
            ctx->port_stats[out_port_idx].num_sent_pkts += sent_cnt;
            ctx->port_stats[out_port_idx].num_sw_drop_pkts += recv_cnt - total_sent_cnt - invalid_cnt;
#endif/*}}}*/

        } // end of rxq scanning
        assert(total_recv_cnt <= NBA_MAX_IO_BATCH_SIZE * ctx->num_hw_rx_queues);
        #ifdef NBA_CPU_MICROBENCH/*{{{*/
        {
            long long ctr[5];
            PAPI_stop(ctx->papi_evset_rx, ctr);
            for (int i = 0; i < 5; i++)
                ctx->papi_ctr_rx[i] += ctr[i];
        }
        #endif/*}}}*/

        while (!rte_ring_empty(ctx->drop_queue)) {
            int n = rte_ring_dequeue_burst(ctx->drop_queue, (void**) drop_pkts,
                                           ctx->num_iobatch_size);
            for (int p = 0; p < n; p++)
                if (drop_pkts[p] != nullptr)
                    rte_pktmbuf_free(drop_pkts[p]);
            ctx->port_stats[0].num_sw_drop_pkts += n;
        }

        /* Scan and execute schedulable elements. */
        ctx->comp_ctx->elem_graph->scan_schedulable_elements(loop_count);

        #ifdef NBA_CPU_MICROBENCH/*{{{*/
        {
            long long ctr[5];
            PAPI_stop(ctx->papi_evset_comp, ctr);
            for (int i = 0; i < 5; i++)
                ctx->papi_ctr_comp[i] += ctr[i];
        }
        #endif/*}}}*/

        while (!rte_ring_empty(ctx->new_packet_request_ring))/*{{{*/
        {
            struct new_packet* new_packet = 0;
            int ret = rte_ring_dequeue(ctx->new_packet_request_ring, (void**) &new_packet);
            assert(ret == 0);

            struct rte_mbuf* pktbuf = rte_pktmbuf_alloc(ctx->new_packet_pool);
            assert(pktbuf != nullptr);

            rte_pktmbuf_pkt_len(pktbuf)  = new_packet->len;
            rte_pktmbuf_data_len(pktbuf) = new_packet->len;
            memcpy(rte_pktmbuf_mtod(pktbuf, void *), new_packet->buf, new_packet->len);

            unsigned txq = ctx->loc.global_thread_idx;
            rte_pktmbuf_free(pktbuf);
            ctx->port_stats[new_packet->out_port].num_sent_pkts++;
            ctx->port_stats[new_packet->out_port].num_sent_bytes += new_packet->len + 24;

            rte_mempool_put(ctx->new_packet_request_pool, new_packet);
        }/*}}}*/

        /* Process received packets. */
        print_ratelimit("# received pkts from all rxq", total_recv_cnt, 10000);
        #ifdef NBA_CPU_MICROBENCH/*{{{*/
        PAPI_start(ctx->papi_evset_comp);
        #endif/*}}}*/
        unsigned comp_batch_size = ctx->comp_ctx->num_combatch_size;
        for (unsigned pidx = 0; pidx < total_recv_cnt; pidx += comp_batch_size) {
            comp_process_batch(ctx, &pkts[pidx], RTE_MIN(comp_batch_size, total_recv_cnt - pidx), loop_count);
        }

        /* The io event loop. */
        if (likely(!ctx->loop_broken))
            ev_run(ctx->loop, EVRUN_NOWAIT);

        loop_count ++;
    }
    if (ctx->loc.local_thread_idx == 0) {
        ctx->init_cond->~CondVar();
        rte_free(ctx->init_cond);
        rte_free(ctx->init_done);
    }
#ifdef TEST_MINIMAL_L2FWD
    rte_free(batch);
#endif
    rte_free(ctx);
    return 0;
}

}

// vim: ts=8 sts=4 sw=4 et foldmethod=marker
