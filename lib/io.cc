/**
 * NBA's RX/TX common components and IO loop
 *
 * Author: Joongi Kim <joongi@an.kaist.ac.kr>
 */

#include "config.hh"
#include "log.hh"
#include "io.hh"
#include "common.hh"
#include "thread.hh"
#include "datablock.hh"
#include "packet.hh"
#include "packetbatch.hh"
/* ===== COMP ===== */
#include "offloadtask.hh"
#include "loadbalancer.hh"
#include "elementgraph.hh"
#include "computecontext.hh"
/* ===== END_OF_COMP ===== */

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
#include <rte_log.h>
#include <rte_memory.h>
#include <rte_memzone.h>
#include <rte_malloc.h>
#include <rte_memcpy.h>
#include <rte_mbuf.h>
#include <rte_byteorder.h>
#include <rte_ethdev.h>
#include <rte_prefetch.h>
#include <rte_cycles.h>
#include <rte_ring.h>
/* {{{ netinet/dpdk header clash resolution */
#ifdef IPPROTO_IP
#undef IPPROTO_IP
#endif
#ifdef IPPROTO_HOPOPTS
#undef IPPROTO_HOPOPTS
#endif
#ifdef IPPROTO_ICMP
#undef IPPROTO_ICMP
#endif
#ifdef IPPROTO_IGMP
#undef IPPROTO_IGMP
#endif
#ifdef IPPROTO_TCP
#undef IPPROTO_TCP
#endif
#ifdef IPPROTO_EGP
#undef IPPROTO_EGP
#endif
#ifdef IPPROTO_PUP
#undef IPPROTO_PUP
#endif
#ifdef IPPROTO_UDP
#undef IPPROTO_UDP
#endif
#ifdef IPPROTO_IDP
#undef IPPROTO_IDP
#endif
#ifdef IPPROTO_TP
#undef IPPROTO_TP
#endif
#ifdef IPPROTO_IPV6
#undef IPPROTO_IPV6
#endif
#ifdef IPPROTO_ROUTING
#undef IPPROTO_ROUTING
#endif
#ifdef IPPROTO_FRAGMENT
#undef IPPROTO_FRAGMENT
#endif
#ifdef IPPROTO_RSVP
#undef IPPROTO_RSVP
#endif
#ifdef IPPROTO_GRE
#undef IPPROTO_GRE
#endif
#ifdef IPPROTO_ESP
#undef IPPROTO_ESP
#endif
#ifdef IPPROTO_AH
#undef IPPROTO_AH
#endif
#ifdef IPPROTO_ICMPV6
#undef IPPROTO_ICMPV6
#endif
#ifdef IPPROTO_NONE
#undef IPPROTO_NONE
#endif
#ifdef IPPROTO_DSTOPTS
#undef IPPROTO_DSTOPTS
#endif
#ifdef IPPROTO_MTP
#undef IPPROTO_MTP
#endif
#ifdef IPPROTO_ENCAP
#undef IPPROTO_ENCAP
#endif
#ifdef IPPROTO_PIM
#undef IPPROTO_PIM
#endif
#ifdef IPPROTO_SCTP
#undef IPPROTO_SCTP
#endif
#ifdef IPPROTO_RAW
#undef IPPROTO_RAW
#endif
/* }}} */
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

#ifdef NBA_SLEEPY_IO
struct rx_state {
    uint16_t rx_length;
    uint16_t rx_quick_sleep;
    uint16_t rx_full_quick_sleep_count;
};
#endif

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
    PacketBatch *b = (PacketBatch*) obj;
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

static inline void comp_check_delayed_operations(comp_thread_context *ctx)
{
    if (likely(!ctx->io_ctx->loop_broken)) {
        ev_run(ctx->io_ctx->loop, EVRUN_NOWAIT);
        ctx->elem_graph->flush_delayed_batches();
        ctx->elem_graph->flush_offloaded_tasks();
    }
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
        OffloadTask *task = tasks[t];
        ComputeContext *cctx = task->cctx;
        #ifdef USE_NVPROF
        nvtxRangePush("task");
        #endif

        assert(task->offload_start == 0);
        task->offload_start = rte_rdtsc();
        /* Run postprocessing handlers. */
        task->postprocess();

        /* Update statistics. */
        float time_spent = (rte_rdtsc() - task->begin_timestamp) / (float) rte_get_tsc_hz();
        uint64_t task_count = ctx->inspector->dev_finished_task_count[task->local_dev_idx];
        ctx->inspector->avg_task_completion_sec[task->local_dev_idx] \
              = (ctx->inspector->avg_task_completion_sec[task->local_dev_idx] * task_count + time_spent) / (task_count + 1);
        ctx->inspector->dev_finished_task_count[task->local_dev_idx] ++;
        ctx->inspector->dev_finished_batch_count[task->local_dev_idx] += task->batches.size();

        /* Enqueue batches for later processing. */
        task->offload_cost += (rte_rdtsc() - task->offload_start);
        task->offload_start = 0;
        double task_time = (task->offload_cost);
        for (size_t b = 0, b_max = task->batches.size(); b < b_max; b ++) {
            task->batches[b]->compute_time += task_time / ((ctx->num_coproc_ppdepth));
            ctx->elem_graph->enqueue_postproc_batch(task->batches[b], task->elem,
                                                    task->input_ports[b]);
        }

        /* Free the task object. */
        task->cctx = nullptr;
        task->~OffloadTask();
        rte_mempool_put(ctx->task_pool, (void *) task);

        /* Free the resources used for this offload task. */
        cctx->currently_running_task = nullptr;
        cctx->state = ComputeContext::READY;
        //ctx->cctx_list.push_back(cctx);

        #ifdef USE_NVPROF
        nvtxRangePop();
        #endif
    }
    #ifdef USE_NVPROF
    nvtxRangePop();
    #endif
}

static void comp_process_batch(io_thread_context *ctx, void *pkts, size_t count, uint64_t loop_count)
{
    assert(count <= ctx->comp_ctx->num_combatch_size);
    if (count == 0) return;
    int ret;
    PacketBatch *batch = NULL;
    do {
        ret = rte_mempool_get(ctx->comp_ctx->batch_pool, (void **) &batch);
        if (ret == -ENOENT) {
            /* Try to free some batches by processing
             * offload-completino events. */
            if (unlikely(ctx->loop_broken))
                break;
            comp_check_delayed_operations(ctx->comp_ctx);
        }
    } while (ret == -ENOENT);
    if (unlikely(ctx->loop_broken))
        return;
    new (batch) PacketBatch();
    memcpy((void **) &batch->packets[0], (void **) pkts, count * sizeof(void*));
    memset(&batch->banno, 0, sizeof(struct annotation_set));

    unsigned p;
    /* t is NOT the actual receive timestamp but a
     * "start-of-processing" timestamp.
     * However its ordering is same as we do FIFO here.
     */
    uint64_t t = rte_rdtsc();
    batch->count = count;
    batch->recv_timestamp = t;
    batch->batch_id = recv_batch_cnt;
    for (p = 0; p < count; p++) {
        batch->excluded[p] = false;
    }
    for (p = 0; p < count; p++) {
        /* Initialize packet metadata objects in pktmbuf's private area. */
        Packet *pkt = Packet::from_base_nocheck(batch->packets[p]);
        new (pkt) Packet(batch, batch->packets[p]);

        /* Set annotations and strip the temporary headroom. */
        pkt->anno.bitmask = 0;
        anno_set(&pkt->anno, NBA_ANNO_IFACE_IN,
                 batch->packets[p]->port);
        anno_set(&pkt->anno, NBA_ANNO_TIMESTAMP, t);
        anno_set(&pkt->anno, NBA_ANNO_BATCH_ID, recv_batch_cnt);
    }
    anno_set(&batch->banno, NBA_BANNO_LB_DECISION, -1);
    recv_batch_cnt ++;

    /* Run the element graph's schedulable elements.
     * FIXME: allow multiple FromInput elements depending on the flow groups. */
    SchedulableElement *input_elem = ctx->comp_ctx->elem_graph->get_entry_point(0);
    PacketBatch *next_batch = nullptr;
    assert(0 != (input_elem->get_type() & ELEMTYPE_INPUT));
    ctx->comp_ctx->input_batch = batch;
    uint64_t next_delay = 0;
    ret = input_elem->dispatch(loop_count, next_batch, next_delay);
    if (next_batch == nullptr) {
        ctx->comp_ctx->elem_graph->free_batch(batch);
    } else {
        assert(next_batch == batch);
        next_batch->has_results = true; // skip processing
        ctx->comp_ctx->elem_graph->run(next_batch, input_elem, 0);
    }
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
    /* Atomically update the counters in the master. */
    ctx->tx_pkt_thruput = 0;
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
            if (!emulate_io && (unsigned) info.pci_dev->numa_node == ctx->loc.node_id) {
                rte_eth_stats_get((uint8_t) j, &s);
                total.port_stats[j].num_rx_drop_pkts = s.ierrors;
            } else {
                total.port_stats[j].num_rx_drop_pkts = 0;
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

static void io_build_packet(char *buf, int size, unsigned flow_idx, random32_func_t rand)/*{{{*/
{
    struct ether_hdr *eth;
    struct ipv4_hdr *ip;
    struct udp_hdr *udp;

    /* Build an ethernet header */
    eth = (struct ether_hdr *)buf;
    eth->ether_type = rte_cpu_to_be_16(ETHER_TYPE_IPv4);

    /* Note: eth->h_source and eth->h_dest are written at send_packets(). */

    /* Build an IPv4 header. */
    ip = (struct ipv4_hdr *)(buf + sizeof(*eth));

    unsigned header_len = 5;
    ip->version_ihl = (4 << 4) | header_len; // header length & version (we are LE!)
    ip->type_of_service = 0;
    ip->total_length = rte_cpu_to_be_16(size - sizeof(*eth));
    ip->packet_id = 0;
    ip->fragment_offset = 0;
    ip->time_to_live = 3;
    ip->next_proto_id = IPPROTO_UDP;
    /* Currently we do not test source-routing. */
    ip->src_addr = rte_cpu_to_be_32(0x0A000001);
    /* Prevent generation of multicast packets, though its probability is very low. */
    if (emulated_num_fixed_flows > 0) {
        ip->dst_addr = rte_cpu_to_be_32(0x0A000000 | (flow_idx & 0x00FFFFFF));
    } else {
        ip->dst_addr = rte_cpu_to_be_32(rand());
        unsigned char *daddr = (unsigned char*)(&ip->dst_addr);
        daddr[0] = 0x0A;
    }
    ip->hdr_checksum = 0;
    ip->hdr_checksum = ip_fast_csum(ip, header_len);

    udp = (struct udp_hdr *)((char *)ip + sizeof(*ip));

    /* For debugging, we fix the source port. */
    udp->src_port = rte_cpu_to_be_16(9999);
    if (emulated_num_fixed_flows > 0) {
        udp->dst_port = rte_cpu_to_be_16(80);
    } else {
        udp->dst_port = rte_cpu_to_be_16((rand() >> 16) & 0xFFFF);
    }

    udp->dgram_len = rte_cpu_to_be_16(size - sizeof(*eth) - sizeof(*ip));
    udp->dgram_cksum = 0;

    /* For debugging, we fill the packet content with a magic number 0xf0. */
    char *content = (char *)((char *)udp + sizeof(*udp));
    memset(content, 0xf0, size - sizeof(*eth) - sizeof(*ip) - sizeof(*udp));
    memset(content, 0xee, 1);  /* To indicate the beginning of packet content area. */
}/*}}}*/

static void io_build_packet_v6(char *buf, int size, unsigned flow_idx, random32_func_t rand)/*{{{*/
{
    struct ether_hdr *eth;
    struct ipv6_hdr *ip;
    struct udp_hdr *udp;

    uint32_t rand_val;

    /* Build an ethernet header. */
    eth = (struct ether_hdr *)buf;
    eth->ether_type = rte_cpu_to_be_16(ETHER_TYPE_IPv6);

    /* Note: eth->h_source and eth->h_dest are written at send_packets(). */

    /* Build an IPv6 header. */
    ip = (struct ipv6_hdr *)(buf + sizeof(*eth));

    ip->vtc_flow = (6 << 4);
    ip->payload_len = rte_cpu_to_be_16(size - sizeof(*eth) - sizeof(*ip));
    ip->hop_limits = 4;
    ip->proto = IPPROTO_UDP;
    /* Currently we do not test source-routing. */
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    ((uint32_t *) ip->src_addr)[0] = rte_cpu_to_be_32(0x0A000001);
    ((uint32_t *) ip->src_addr)[1] = rte_cpu_to_be_32(0x00000000);
    ((uint32_t *) ip->src_addr)[2] = rte_cpu_to_be_32(0x00000000);
    ((uint32_t *) ip->src_addr)[3] = rte_cpu_to_be_32(0x00000000);
    ((uint32_t *) ip->dst_addr)[0] = rte_cpu_to_be_32(rand());
    ((uint32_t *) ip->dst_addr)[1] = rte_cpu_to_be_32(rand());
    ((uint32_t *) ip->dst_addr)[2] = rte_cpu_to_be_32(rand());
    ((uint32_t *) ip->dst_addr)[3] = rte_cpu_to_be_32(rand());
    #pragma GCC diagnostic pop
    /* Prevent generation of multicast packets. */
    ip->dst_addr[0] = 0x0A;

    // TODO: support fixed flows in IPv6 as well.

    udp = (struct udp_hdr *)((char *)ip + sizeof(*ip));

    rand_val = rand();
    udp->src_port = rte_cpu_to_be_16(rand_val & 0xFFFF);
    udp->dst_port = rte_cpu_to_be_16((rand_val >> 16) & 0xFFFF);

    udp->dgram_len = rte_cpu_to_be_16(size - sizeof(*eth) - sizeof(*ip));
    udp->dgram_cksum = 0;

    /* For debugging, we fill the packet content with a magic number 0xf0. */
    char *content = (char *)((char *)udp + sizeof(*udp));
    memset(content, 0xf0, size - sizeof(*eth) - sizeof(*ip) - sizeof(*udp));
    memset(content, 0xee, 1);  /* To indicate the beginning of packet content area. */
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
    PacketBatch out_batches[NBA_MAX_PORTS];
    uint64_t t = rte_rdtsc();
    ctx->comp_ctx->inspector->batch_process_time = 0.01 * (t - batch->recv_timestamp)
                                                   + 0.99 * ctx->comp_ctx->inspector->batch_process_time;
    unsigned p;
    {
        int64_t banno = anno_get(&batch->banno, NBA_BANNO_LB_DECISION);
        double prev_pkt_time = 0;
        double true_time = batch->compute_time;
        if (banno == -1) {
            prev_pkt_time = ctx->comp_ctx->inspector->true_process_time_cpu;
            prev_pkt_time = (((prev_pkt_time * (CPU_HISTORY_SIZE - 1)) + true_time) / CPU_HISTORY_SIZE);
            ctx->comp_ctx->inspector->true_process_time_cpu = prev_pkt_time;
        } else {
            prev_pkt_time = ctx->comp_ctx->inspector->true_process_time_gpu[banno];
            prev_pkt_time = (((prev_pkt_time * (GPU_HISTORY_SIZE - 1)) + true_time) / GPU_HISTORY_SIZE);
            ctx->comp_ctx->inspector->true_process_time_gpu[banno] = prev_pkt_time;
        }
    }
//#ifdef NBA_CPU_MICROBENCH
//    PAPI_start(ctx->papi_evset_tx);
//#endif

    // TODO: keep ordering of packets (or batches)
    //   NOTE: current implementation: no extra queueing,
    //   just transmit as requested
    for (p = 0; p < batch->count; p++) {
        Packet *pkt = Packet::from_base(batch->packets[p]);
        if (batch->excluded[p] == false && anno_isset(&pkt->anno, NBA_ANNO_IFACE_OUT)) {
            struct ether_hdr *ethh = rte_pktmbuf_mtod(batch->packets[p], struct ether_hdr *);
            uint64_t o = anno_get(&pkt->anno, NBA_ANNO_IFACE_OUT);

            /* Update source/dest MAC addresses. */
            ether_addr_copy(&ethh->s_addr, &ethh->d_addr);
            ether_addr_copy(&ctx->tx_ports[o].addr, &ethh->s_addr);

            /* Append to the corresponding output batch. */
            int cnt = out_batches[o].count ++;
            out_batches[o].packets[cnt] = batch->packets[p];
        } else {
            assert(batch->packets[p] == nullptr);
        }
    }

    unsigned tx_tries = 0;
    for (unsigned o = 0; o < ctx->num_tx_ports; o++) {
        if (out_batches[o].count == 0)
            continue;
        struct rte_mbuf **pkts = out_batches[o].packets;
        unsigned count = out_batches[o].count;
        uint64_t total_sent_byte = 0;

        /* Sum TX packet bytes. */
        for(unsigned k = 0; k < count; k++) {
            struct rte_mbuf* cur_pkt = out_batches[o].packets[k];
            unsigned len = rte_pktmbuf_pkt_len(cur_pkt) + 24;  /* Add Ethernet overheads */
            ctx->port_stats[o].num_sent_bytes += len;
        }

        if (ctx->mode == IO_EMUL) {
            /* Emulated TX always succeeds without drops. */
            rte_mempool_put_bulk(ctx->emul_rx_packet_pool, (void **) pkts, count);
            ctx->port_stats[o].num_sent_pkts += count;
            ctx->global_tx_cnt += count;
            ctx->port_stats[o].num_tx_drop_pkts += 0;
        } else {
#if NBA_OQ
            /* To implement output-queuing, we need to drop when the TX NIC
             * is congested.  This would not happen in high line rates such
             * as 10 GbE because processing speed becomes the bottleneck,
             * but it will be meaningful when we use low-speed NICs such as
             * 1 GbE cards. */
            unsigned txq = ctx->loc.global_thread_idx;
            unsigned sent_cnt = rte_eth_tx_burst((uint8_t) o, txq, pkts, count);
            for (unsigned k = sent_cnt; k < count; k++)
                rte_pktmbuf_free(pkts[k]);
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
    }
    double &thruput = ctx->comp_ctx->inspector->tx_pkt_thruput;
    ctx->LB_THRUPUT_WINDOW_SIZE = 16384; // (1 << 16);
    uint64_t tick = rte_rdtsc_precise();
    //if (tick - ctx->last_tx_tick > rte_get_tsc_hz() * 0.1) {
        double thr = (double) ctx->global_tx_cnt * 1e3 / (tick - ctx->last_tx_tick);
        thruput = (thruput * (ctx->LB_THRUPUT_WINDOW_SIZE - 1) + thr) / ctx->LB_THRUPUT_WINDOW_SIZE;
        ctx->global_tx_cnt = 0;
        ctx->last_tx_tick = tick;
    //}
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

    /* Packet builder initialization for emulation mode. */
    random32_func_t random32 = bind(uniform_int_distribution<uint32_t>{}, mt19937());
    packet_builder_func_t build_packet = (ctx->emul_ip_version == 4) ?
                                         io_build_packet : io_build_packet_v6;

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
        if (ctx->mode == IO_EMUL) {
            struct ether_addr dummy_addr = {{0x4e, 0x42, 0x41, 0x00, 0x00, (uint8_t) i}};
            ether_addr_copy(&dummy_addr, &ctx->tx_ports[i].addr);
        } else {
            rte_eth_macaddr_get(i, &ctx->tx_ports[i].addr);
        }
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
                                                   sizeof(PacketBatch), 0, //(unsigned) (ctx->comp_ctx->num_batchpool_size / 1.5),
                                                   0, nullptr, nullptr,
                                                   comp_packetbatch_init, nullptr,
                                                   ctx->loc.node_id, 0);
    if (ctx->comp_ctx->batch_pool == nullptr)
        rte_panic("RTE_ERROR while creating comp_ctx->batch_pool: %s\n", rte_strerror(rte_errno));

    snprintf(temp, RTE_MEMPOOL_NAMESIZE,
        "comp.dbstate.%u:%u@%u", ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
    size_t dbstate_pool_size = NBA_MAX_COPROC_PPDEPTH;
    size_t dbstate_item_size = sizeof(struct datablock_tracker) * NBA_MAX_DATABLOCKS;
    ctx->comp_ctx->dbstate_pool = rte_mempool_create(temp, dbstate_pool_size + 1,
                                                     dbstate_item_size, 0, //(unsigned) (dbstate_pool_size / 1.5),
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
                                                  sizeof(OffloadTask), 0, //(unsigned) (ctx->comp_ctx->num_taskpool_size / 1.5),
                                                  0, nullptr, nullptr,
                                                  comp_task_init, nullptr,
                                                  ctx->loc.node_id, 0);
    if (ctx->comp_ctx->task_pool == nullptr)
        rte_panic("RTE_ERROR while creating comp_ctx->task pool: %s\n", rte_strerror(rte_errno));

    ctx->comp_ctx->packet_pool = packet_create_mempool(128, ctx->loc.node_id, ctx->loc.core_id);
    assert(ctx->comp_ctx->packet_pool != nullptr);

    ctx->comp_ctx->inspector = (SystemInspector *) rte_malloc_socket(NULL, sizeof(SystemInspector),
                                                                     CACHE_LINE_SIZE, ctx->loc.node_id);
    new (ctx->comp_ctx->inspector) SystemInspector();

    /* Register the offload completion event. */
    if (ctx->comp_ctx->coproc_ctx != nullptr) {
        ev_async_init(ctx->comp_ctx->task_completion_watcher, comp_offload_task_completion_cb);
        // TODO: remove this event and just check the completion queue on every iteration.
        ev_async_start(ctx->loop, ctx->comp_ctx->task_completion_watcher);
    }
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

    uint64_t cur_tsc, prev_tsc = 0;
#ifdef NBA_SLEEPY_IO
    struct rx_state *states = (struct rx_state *) rte_malloc_socket("rxstates",
            sizeof(*states) * ctx->num_hw_rx_queues,
            64, ctx->loc.node_id);
    memset(states, 0, sizeof(*states) * ctx->num_hw_rx_queues);
#endif
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
    if (ctx->mode == IO_EMUL) {
        /* Create random packets. */
        if (emulated_num_fixed_flows > 0)
            num_random_packets = RTE_MIN(emulated_num_fixed_flows, num_random_packets);
        for (int i = 0; i < num_random_packets; i++) {
            random_packets[i] = (char *) rte_zmalloc_socket("randpkt", ctx->emul_packet_size,
                                                            CACHE_LINE_SIZE, ctx->loc.node_id);
            assert(random_packets[i] != nullptr);
            build_packet(random_packets[i], ctx->emul_packet_size, (emulated_num_fixed_flows == 0) ? 0 : i, random32);
            struct ether_hdr *ethh = (struct ether_hdr *) random_packets[i];
            eth_random_addr(&ethh->s_addr.addr_bytes[0]);
        }
    }

    // ctx->num_iobatch_size = 1; // FOR TESTING
    uint64_t loop_count = 0;
    ctx->last_tx_tick = rte_rdtsc();

    /* The IO thread runs in polling mode. */
    while (likely(!ctx->loop_broken)) {
        unsigned total_recv_cnt = 0;
        #ifdef NBA_CPU_MICROBENCH
        PAPI_start(ctx->papi_evset_rx);
        #endif
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
            cur_tsc = rte_rdtsc();

            if (ctx->mode == IO_EMUL) {/*{{{*/
                ret = rte_mempool_get_bulk(ctx->emul_rx_packet_pool, (void **) &pkts[total_recv_cnt], ctx->num_iobatch_size);
                if (ret == 0) {
                    #define PREFETCH_MAX (4)
                    #if PREFETCH_MAX
                    for (signed p = 0; p < RTE_MIN(PREFETCH_MAX, ((signed)ctx->num_iobatch_size)); p++)
                        rte_prefetch0((char * ) pkts[total_recv_cnt + p]->buf_addr + RTE_PKTMBUF_HEADROOM);
                    #endif
                    for (j = total_recv_cnt; j < total_recv_cnt + ctx->num_iobatch_size; j++) {
                        pkts[j]->packet_type = 0;
                        pkts[j]->data_off    = RTE_PKTMBUF_HEADROOM;
                        rte_pktmbuf_pkt_len(pkts[j])  = ctx->emul_packet_size;
                        rte_pktmbuf_data_len(pkts[j]) = ctx->emul_packet_size;
                        pkts[j]->port    = port_idx;
                        pkts[j]->nb_segs = 1;
                        pkts[j]->next    = nullptr;
                        rte_mbuf_refcnt_set(pkts[j], 1);

                        /* Copy the content from pre-created random packets. */
                        rte_memcpy(rte_pktmbuf_mtod(pkts[j], char *), random_packets[rp], ctx->emul_packet_size);

                        struct ether_hdr *ethh = rte_pktmbuf_mtod(pkts[j], struct ether_hdr *);
                        ether_addr_copy(&ctx->tx_ports[port_idx].addr, &ethh->d_addr);

                        struct ipv4_hdr *iph = (struct ipv4_hdr *) (ethh + 1);
                        assert(iph->time_to_live > 1);

                        #if PREFETCH_MAX
                        if (j + PREFETCH_MAX + 1 < total_recv_cnt + ctx->num_iobatch_size)
                            rte_prefetch0((char * ) pkts[j + 1]->buf_addr + RTE_PKTMBUF_HEADROOM);
                        #endif
                        rp = (rp + 1) % num_random_packets;
                    }
                    #undef PREFETCH_MAX
                    //printf("emulrxpktpool got %u / %u\n", ctx->num_iobatch_size, rte_mempool_count(ctx->emul_rx_packet_pool));
                    recv_cnt = ctx->num_iobatch_size;
                } else {
                    /* If no emulated packets are available, break out the rxq scan loop.
                     * (This corresponds to "idle" state with real traffic,
                     * but for maximum-performance testing this case should not happen.) */
                    printf("emulrxpktpool no?? %u\n", rte_mempool_count(ctx->emul_rx_packet_pool));
                    assert(0);
                    break;
                }
            } else {/*}}}*/
#ifdef NBA_SLEEPY_IO /*{{{*/
                struct rx_state *state = &states[i];
                if ((state->rx_quick_sleep --) > 0) {
                    pthread_yield();
#  if !defined(TEST_RXONLY) && !defined(TEST_MINIMAL_L2FWD)
                    goto __skip_rx;
#  else
                    continue;
#  endif
                }

#  if !defined(TEST_RXONLY) && !defined(TEST_MINIMAL_L2FWD)
                state->rx_length = rte_eth_rx_burst((uint8_t) port_idx, rxq,
                                                    //pkts, RTE_MIN(free_cnt, ctx->num_iobatch_size));
                                                    &pkts[total_recv_cnt], ctx->num_iobatch_size);
#  else
                state->rx_length = rte_eth_rx_burst((uint8_t) port_idx, rxq,
                                                    &pkts[total_recv_cnt], ctx->num_iobatch_size);
#  endif
                state->rx_quick_sleep = (uint16_t) (ctx->num_iobatch_size - state->rx_length);
                if (state->rx_length != 0) {
                    /* We received some packets, and expect some more packets may arrive soon.
                     * Reset the sleep amount. */
                    state->rx_full_quick_sleep_count = 0;
                } else {
                    /* No packets. We need some sleep. Adjust how much to sleep. */
                    if (state->rx_full_quick_sleep_count < ctx->num_iobatch_size)
                        /* Increase sleep count if no return happens repeatedly. */
                        state->rx_full_quick_sleep_count ++;
                    state->rx_quick_sleep = (uint16_t) (state->rx_quick_sleep \
                                                        * state->rx_full_quick_sleep_count);
                }
                recv_cnt = state->rx_length;
#else /*}}}*/
                recv_cnt = rte_eth_rx_burst((uint8_t) port_idx, rxq,
                                             &pkts[total_recv_cnt], ctx->num_iobatch_size);
#endif        /* endif NBA_SLEEPY_IO */
            } /* endif IO_EMUL */

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

            prev_tsc = cur_tsc;

        } // end of rxq scanning
        assert(total_recv_cnt <= NBA_MAX_IO_BATCH_SIZE * ctx->num_hw_rx_queues);
        #ifdef NBA_CPU_MICROBENCH
        {
            long long ctr[5];
            PAPI_stop(ctx->papi_evset_rx, ctr);
            for (int i = 0; i < 5; i++)
                ctx->papi_ctr_rx[i] += ctr[i];
        }
        #endif

        if (ctx->mode == IO_EMUL) {/*{{{*/
            while (!rte_ring_empty(ctx->drop_queue)) {
                int n = rte_ring_dequeue_burst(ctx->drop_queue, (void**) drop_pkts,
                                               ctx->num_iobatch_size);
                assert(n >= 0);
                rte_mempool_put_bulk(ctx->emul_rx_packet_pool, (void **) drop_pkts, n);
                ctx->port_stats[0].num_sw_drop_pkts += n;
            }
        } else {/*}}}*/
            while (!rte_ring_empty(ctx->drop_queue)) {
                int n = rte_ring_dequeue_burst(ctx->drop_queue, (void**) drop_pkts,
                                               ctx->num_iobatch_size);
                assert(n >= 0);
                for (int p = 0; p < n; p++)
                    rte_pktmbuf_free(drop_pkts[p]);
                ctx->port_stats[0].num_sw_drop_pkts += n;
            }
        }

        /* Process received packets. */
        print_ratelimit("# received pkts from all rxq", total_recv_cnt, 10000);

        ctx->comp_ctx->stop_task_batching = (total_recv_cnt == 0); // not used currently...
        #ifdef NBA_CPU_MICROBENCH
        PAPI_start(ctx->papi_evset_comp);
        #endif

        ctx->comp_ctx->elem_graph->flush_offloaded_tasks();
        ctx->comp_ctx->elem_graph->flush_delayed_batches();
        unsigned comp_batch_size = ctx->comp_ctx->num_combatch_size;
        for (unsigned pidx = 0; pidx < total_recv_cnt; pidx += comp_batch_size) {
            comp_process_batch(ctx, &pkts[pidx], RTE_MIN(comp_batch_size, total_recv_cnt - pidx), loop_count);
        }

        const auto &selems = ctx->comp_ctx->elem_graph->get_schedulable_elements();
        for (SchedulableElement *selem : selems) {
            PacketBatch *next_batch = nullptr;
            if (0 == (selem->get_type() & ELEMTYPE_INPUT)) { // FromInput is executed in comp_process_batch() already.
                while (true) {
                    /* Try to "drain" internally stored batches. */
                    if (selem->_last_delay == 0) {
                        ret = selem->dispatch(loop_count, next_batch, selem->_last_delay);
                        if (selem->_last_check_tick == 0)
                            selem->_last_call_ts = get_usec();
                        selem->_last_check_tick ++;
                    } else if (selem->_last_check_tick > 100) { // rate-limit calls to clock_gettime()
                        uint64_t now = get_usec();
                        if (now >= selem->_last_call_ts + selem->_last_delay) {
                            ret = selem->dispatch(loop_count, next_batch, selem->_last_delay);
                            selem->_last_call_ts = now;
                        }
                        selem->_last_check_tick = 0;
                    } else
                        selem->_last_check_tick ++;
                    if (next_batch != nullptr) {
                        next_batch->has_results = true; // skip processing
                        ctx->comp_ctx->elem_graph->run(next_batch, selem, 0);
                    } else
                        break;
                };
            }
        }
        #ifdef NBA_CPU_MICROBENCH
        {
            long long ctr[5];
            PAPI_stop(ctx->papi_evset_comp, ctr);
            for (int i = 0; i < 5; i++)
                ctx->papi_ctr_comp[i] += ctr[i];
        }
        #endif

        loop_count ++;

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
            if (ctx->mode != IO_EMUL) {
                if(0 == rte_eth_tx_burst(new_packet->out_port, txq, &pktbuf, 1)) {
                    RTE_LOG(DEBUG, IO, "tx failed.\n");
                    rte_pktmbuf_free(pktbuf);
                    ctx->port_stats[new_packet->out_port].num_tx_drop_pkts++;
                } else {
                    ctx->port_stats[new_packet->out_port].num_sent_pkts++;
                    ctx->port_stats[new_packet->out_port].num_sent_bytes += new_packet->len + 24;
                }
            } else {
                rte_pktmbuf_free(pktbuf);
                ctx->port_stats[new_packet->out_port].num_sent_pkts++;
                ctx->port_stats[new_packet->out_port].num_sent_bytes += new_packet->len + 24;
            }

            rte_mempool_put(ctx->new_packet_request_pool, new_packet);
        }/*}}}*/

        /* The io event loop. */
        if (likely(!ctx->loop_broken))
            ev_run(ctx->loop, EVRUN_NOWAIT);
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
