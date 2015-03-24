/**
 * NBA's EMMA architecture main program.
 *
 * Author: Joongi Kim <joongi@an.kaist.ac.kr>
 */

#define _POSIX_C_SOURCE 2

#include "lib/collision_checker.h"
#include "lib/config.hh"
#include "lib/common.hh"
#include "lib/log.hh"
#include "lib/thread.hh"
#include "lib/io.hh"
#include "lib/computedevice.hh"
#include "lib/computation.hh"
#include "lib/coprocessor.hh"
#include "lib/annotation.hh"
#include "lib/strutils.hh"
#include "lib/queue.hh"
#include "lib/nodelocalstorage.hh"
#include "lib/memory_space.hh"
#ifdef USE_CUDA
#include "engines/cuda/utils.hh"
#include "engines/cuda/computedevice.hh"
#include "engines/cuda/computecontext.hh"
#include <cuda.h>
#endif
#ifdef USE_PHI
#include "engines/phi/utils.hh"
#include "engines/phi/computedevice.hh"
#include "engines/phi/computecontext.hh"
#endif
#include "engines/dummy/computedevice.hh"
#include "engines/dummy/computecontext.hh"

#include <set>
#include <string>
#include <cstring>
#include <cstdint>
#include <cstdbool>
#include <vector>
#include <unordered_set>
#include <map>
#include <thread>
extern "C" {
//#include <papi.h>
#include <unistd.h>
#include <limits.h>
#include <signal.h>
#include <pthread.h>
#include <numa.h>
#include <locale.h>
#include <getopt.h>
#include <sys/prctl.h>
#include <rte_config.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_log.h>
#include <rte_memory.h>
#include <rte_malloc.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_tailq.h>
#include <rte_prefetch.h>
#include <rte_branch_prediction.h>
#include <rte_pci.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_ring.h>
}

/* According to dpdk-dev mailing list,
 * NUM_MBUF for the whole system should be greater than:
 *   (hw-rx-ring-size * nb-rx-queue) + (hw-tx-ring-size * nb-tx-queue)
 *   + (nb-lcores * mbuf-pool-cache-size)
 */
#define NUM_MBUF    (8192)

using namespace std;
using namespace nba;

static unsigned num_nodes, num_coprocessors, num_pcores, num_lcores, num_ports;
static unsigned num_coproc_threads, num_comp_threads, num_io_threads;
static struct spawned_thread *coprocessor_threads;
static struct spawned_thread *computation_threads;
static struct spawned_thread *io_threads;

static CondVar _exit_cond;
static bool _terminated = false;
static thread_id_t main_thread_id;

struct mbuf_ctor_arg {
	uint16_t seg_buf_offset; /**< offset of data in data segment of mbuf. */
	uint16_t seg_buf_size;   /**< size of data segment in mbuf. */
};

struct mbuf_pool_ctor_arg {
	uint16_t seg_buf_size; /**< size of data segment in mbuf. */
};

static void handle_signal(int signum);

static void invalid_cb(struct ev_loop *loop, struct ev_async *w, int revents)
{
    rte_panic("BUG: Callback was not set!!\n");
}

static void pktmbuf_pool_ctor(struct rte_mempool *mp, void *opaque_arg)
{
    struct mbuf_pool_ctor_arg       *mbp_ctor_arg;
    struct rte_pktmbuf_pool_private *mbp_priv;

    mbp_ctor_arg = (struct mbuf_pool_ctor_arg *) opaque_arg;
    mbp_priv = (struct rte_pktmbuf_pool_private *) rte_mempool_get_priv(mp);
    mbp_priv->mbuf_data_room_size = mbp_ctor_arg->seg_buf_size;
}

static void pktmbuf_obj_ctor(struct rte_mempool *mp, void *opaque_arg, void *raw_mbuf, unsigned idx)
{
    struct mbuf_ctor_arg *mb_ctor_arg;
    struct rte_mbuf      *mb;

    mb_ctor_arg = (struct mbuf_ctor_arg *) opaque_arg;
    mb = (struct rte_mbuf *) raw_mbuf;

    mb->packet_type  = 0;
    mb->pool         = mp;
    mb->buf_addr     = (void *) ((char *)mb + mb_ctor_arg->seg_buf_offset);
    mb->buf_physaddr = (uint64_t) (rte_mempool_virt2phy(mp, mb) +
                                   mb_ctor_arg->seg_buf_offset);
    mb->buf_len          = mb_ctor_arg->seg_buf_size;
    mb->ol_flags         = 0;
    mb->data_off         = RTE_PKTMBUF_HEADROOM;
    mb->nb_segs          = 1;
    mb->hash.rss         = 0;
}

int main(int argc, char **argv)
{
    //prevent multiple processes
    printf("Trying to acquire a process lock...\n");
    fflush(stdout);
    int collision_flag = 0;
    if(geteuid() != 0)
    {
    	printf("NBA is running on USER privilege.\n");
    	printf("Using tmp directory for checking collisions.\n");
    	collision_flag |= COLLISION_USE_TEMP;
    }
    else
    {
    	printf("NBA is running on ROOT privilege.\n");
    }
    if(0 != check_collision("NBA", collision_flag))
    {
        printf("Cannot acquire a process lock. Exiting.\n");
        fflush(stdout);
        return 1;
    }
    printf("Lock acquired!\n");
    fflush(stdout);

    int ret;
    unsigned loglevel = RTE_LOG_INFO;
    unsigned i, j, k;
    uint8_t port_idx, ring_idx;
    unsigned num_rxq_per_port, num_txq_per_port;
    unsigned num_rx_attached_layers;

    /* Prepare to spawn worker threads. */
    main_thread_id = threading::self();

    struct {
        struct port_info rx_ports[NBA_MAX_PORTS];
        unsigned num_rx_ports;
    } node_ports[NBA_MAX_NODES];

    //assert(PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT);
    //assert(PAPI_thread_init(pthread_self) == PAPI_OK);
    setlocale(LC_NUMERIC, "");

    /* Initialize DPDK EAL and move argument pointers. */
    rte_set_application_usage_hook([] (const char *prgname) {
        printf("Usage: %s [EAL options] -- [--emulate-io=NUM] ... [-l LEVEL] <system-config-path> <pipeline-config-path>\n\n", prgname);
        printf("NBA options:\n");
        printf("  --dummy-device             : Use dummy offloading devices. (default: false)\n");
        printf("  --emulate-io[=NUM]         : Use random-generated packets instead of\n"
               "                               receiving packets from NICs. Transmitted packets\n"
               "                               are silently dropped.  The default NUM_NICS is 8.\n");
        printf("  --emulated-pktsize=NUM     : The size of packets for emulation. (default: 64)\n");
        printf("  --emulated-ipversion=NUM   : The IP version of packets for emulation. (default: 4)\n");
        printf("  --emulated-fixed-flows=NUM : The number of fixed flows. If zero, all packets are randomized.\n"
               "                               (default: 0) \n");
        printf("  -l, --loglevel=[LEVEL]     : The log level to control output verbosity.\n"
               "                               The default is \"info\".  Available values are:\n"
               "                               debug, info, notice, warning, error, critical, alert, emergency.\n");
    });
    /* At this moment, we cannot customize log level because we haven't
     * parsed the arguments yet. */
    rte_set_log_level(RTE_LOG_INFO);
    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Invalid EAL parameters.\n");
    argc -= ret;
    argv += ret;

    /* Parse command-line arguments. */
    dummy_device = false;
    emulate_io = false;
    emulated_ip_version = 4;
    emulated_packet_size = 64;
    emulated_num_fixed_flows = 0;
    num_emulated_ifaces = 8;
    char *system_config = new char[PATH_MAX];
    char *pipeline_config = new char[PATH_MAX];

    struct option long_opts[] = {
        {"dummy-device", optional_argument, NULL, 0},
        {"emulate-io", optional_argument, NULL, 0},
        {"emulated-pktsize", required_argument, NULL, 0},
        {"emulated-ipversion", required_argument, NULL, 0},
        {"emulated-fixed-flows", required_argument, NULL, 0},
        {"loglevel", required_argument, NULL, 'l'},
        {0, 0, 0, 0}
    };
    while (true) {
        int optidx = 0;
        int c = getopt_long(argc, argv, "l:", long_opts, &optidx);
        if (c == -1) break;
        switch (c) {
        case 0:
            /* Skip flag-only options. */
            if (long_opts[optidx].flag != NULL) break;
            /* Process {long_opts[optidx].name}:{optarg} kv pairs. */
            if (!strcmp("dummy-device", long_opts[optidx].name)) {
                dummy_device = true;
            } else if (!strcmp("emulate-io", long_opts[optidx].name)) {
                emulate_io = true;
                if (optarg)
                    num_emulated_ifaces = (unsigned) atol(optarg);
                assert(num_emulated_ifaces > 0);
                assert(num_emulated_ifaces <= NBA_MAX_PORTS);
            }
            else if (!strcmp("emulated-pktsize", long_opts[optidx].name)) {
                emulated_packet_size = (unsigned) atol(optarg);
                assert(emulated_packet_size >= 64);
                assert(emulated_packet_size <= 1514);
            }
            else if (!strcmp("emulated-ipversion", long_opts[optidx].name)) {
                emulated_ip_version = (int) atol(optarg);
                assert(emulated_ip_version == 4 || emulated_ip_version == 6);
            }
            else if (!strcmp("emulated-fixed-flows", long_opts[optidx].name)) {
                emulated_num_fixed_flows = (int) atol(optarg);
            }
            break;
        case 'l':
            assert(optarg != NULL);
            if (!strcmp("debug", optarg))
                loglevel = RTE_LOG_DEBUG;
            else if (!strcmp("info", optarg))
                loglevel = RTE_LOG_INFO;
            else if (!strcmp("notice", optarg))
                loglevel = RTE_LOG_NOTICE;
            else if (!strcmp("warning", optarg))
                loglevel = RTE_LOG_WARNING;
            else if (!strcmp("error", optarg))
                loglevel = RTE_LOG_ERR;
            else if (!strcmp("critical", optarg))
                loglevel = RTE_LOG_CRIT;
            else if (!strcmp("emergency", optarg))
                loglevel = RTE_LOG_EMERG;
            else
                rte_exit(EXIT_FAILURE, "Invalid value for loglevel: %s\n", optarg);
            break;
        case '?':
            /* getopt has already printed the error message. */
            break;
        default:
            rte_exit(EXIT_FAILURE, "Invalid NBA arguments.\n");
        }
    }
    if (optind + 2 < argc - 1) {
        printf("You need at least two positional arguments:\n"
               "  main [EAL options] -- [NBA options] <system-config-path> <pipeline-config-path>\n");
        rte_exit(EXIT_FAILURE, "Not enough NBA arguments.\n");
    }
    for (int optidx = optind; optidx < argc; optidx++) {
        switch (optidx - optind) {
        case 0:
            strncpy(system_config, argv[optidx], PATH_MAX);
            break;
        case 1:
            strncpy(pipeline_config, argv[optidx], PATH_MAX);
            break;
        default:
            rte_exit(EXIT_FAILURE, "Too many NBA arguments.\n");
        }
    }
    RTE_LOG(INFO, MAIN, "Setting log level to %d.\n", loglevel);
    rte_set_log_level(loglevel);

    /* NOTE: EAL spawns its own worker threads which is not used. */
    rte_set_log_type(RTE_LOGTYPE_PMD, false);
    rte_set_log_type(RTE_LOGTYPE_MALLOC, false);
    rte_set_log_type(RTE_LOGTYPE_MEMPOOL, false);

    if (emulate_io) {
        num_ports = num_emulated_ifaces;
        RTE_LOG(NOTICE, MAIN, "IO emulation mode enabled!\n");
        RTE_LOG(INFO, MAIN, "  Emulated IP version: %d\n", emulated_ip_version);
        RTE_LOG(INFO, MAIN, "  Emulated packet size: %d\n", emulated_packet_size);
        if (emulated_num_fixed_flows > 0)
            RTE_LOG(INFO, MAIN, "  Using fixed flows: %d\n", emulated_num_fixed_flows);
    } else {
        num_ports = rte_eth_dev_count();
    }
    RTE_LOG(NOTICE, MAIN, "Detected %u ports.\n", num_ports);
    if (num_ports == 0)
        rte_exit(EXIT_FAILURE, "No available/compatible ehternet ports.\n");

    threading::bind_cpu(0);

    num_lcores = sysconf(_SC_NPROCESSORS_ONLN);
    num_nodes = numa_num_configured_nodes();
    bool numa_disabled = num_nodes == 1;
    if(numa_disabled)
    	printf("NUMA is disabled.\n");
    else
    	printf("%d NUMA nodes are enabled.\n", num_nodes);
    num_pcores = check_ht_enabled() ? num_lcores / 2 : num_lcores;

    /* We have two types of configurations: system and Click.
     *
     * The system configuration includes:
     *  - variable parameters such as maximum queue lengths, maximum
     *    batch sizes, etc.
     *  - the number of threads for IO, computation, and corpocessor
     *    handling
     *  - the mapping of threads and CPU cores
     *  - the connections between IO-computation threads and
     *    computation-coprocessor threads
     *
     * The Click configuration includes:
     *  - what elements are used in the processing pipeline
     *  - the connections between elements represeting the order of
     *    execution
     * It is written in the Click configuration language.
     */

    /* Read configuration. */
    RTE_LOG(INFO, MAIN, "Loading system configuration from \"%s\"...\n", system_config);
    RTE_LOG(WARNING, MAIN, "If it hangs, try to restart hold_gpu script. (cuda service if installed as upstart)\n");
    if (!load_config(system_config)) {
        rte_exit(EXIT_FAILURE, "Loading global configuration has failed.\n");
    }
    if (num_ports > NBA_MAX_PORTS)
        num_ports = NBA_MAX_PORTS;

    num_rxq_per_port = system_params["NUM_RXQ_PER_PORT"];
    num_txq_per_port = num_lcores;
    RTE_LOG(DEBUG, MAIN, "num_rxq_per_port = %u, num_txq_per_port = %u\n", num_rxq_per_port, num_txq_per_port);

    //RTE_LOG(INFO, MAIN, "Reading pipeline configuration from \"%s\"...\n", pipeline_config);
    //conf_graph = create_graph(pipeline_config);
    //if (conf_graph.size() == 0) {
    //    rte_exit(EXIT_FAILURE, "Could not open the pipeline configuration.\n");
    //}

    /* Prepare per-port configurations. */
    struct rte_eth_conf port_conf;
    memset(&port_conf, 0, sizeof(port_conf));
    port_conf.rxmode.mq_mode        = ETH_RSS;

    uint8_t hash_key[40];
    for (unsigned k = 0; k < sizeof(hash_key); k++)
        hash_key[k] = (uint8_t) rand();
    port_conf.rx_adv_conf.rss_conf.rss_key = hash_key;
    port_conf.rx_adv_conf.rss_conf.rss_hf = ETH_RSS_IPV4_TCP | ETH_RSS_IPV4_UDP | ETH_RSS_IPV6_TCP | ETH_RSS_IPV6_UDP;
    port_conf.rxmode.max_rx_pkt_len = 0; /* only used if jumbo_frame is enabled */
    port_conf.rxmode.split_hdr_size = 0;
    port_conf.rxmode.header_split   = false;
    port_conf.rxmode.hw_ip_checksum = false;
    port_conf.rxmode.hw_vlan_filter = false;
    port_conf.rxmode.hw_vlan_strip  = false;
    port_conf.rxmode.hw_vlan_extend = false;
    port_conf.rxmode.jumbo_frame    = false;
    port_conf.rxmode.hw_strip_crc   = true;
    port_conf.txmode.mq_mode    = ETH_DCB_NONE;
    port_conf.fdir_conf.mode    = RTE_FDIR_MODE_NONE;
    port_conf.fdir_conf.pballoc = RTE_FDIR_PBALLOC_64K;
    port_conf.fdir_conf.status  = RTE_FDIR_NO_REPORT_STATUS;
    port_conf.fdir_conf.flexbytes_offset = 0;
    port_conf.fdir_conf.drop_queue       = 0;

    /* Per RX-queue configuration */
    struct rte_eth_rxconf rx_conf;
    memset(&rx_conf, 0, sizeof(rx_conf));
    rx_conf.rx_thresh.pthresh = 8;
    rx_conf.rx_thresh.hthresh = 4;
    rx_conf.rx_thresh.wthresh = 4;
    rx_conf.rx_free_thresh = 32;
    rx_conf.rx_drop_en     = 0; /* when enabled, drop packets if no descriptors are available */
    const unsigned num_rx_desc = system_params["IO_RXDESC_PER_HWRXQ"];

    /* Per TX-queue configuration */
    struct rte_eth_txconf tx_conf;
    memset(&tx_conf, 0, sizeof(tx_conf));
    tx_conf.tx_thresh.pthresh = 36;
    tx_conf.tx_thresh.hthresh = 4;
    tx_conf.tx_thresh.wthresh = 0;
    /* The following rs_thresh and flag value enables "simple TX" function. */
    tx_conf.tx_rs_thresh   = 32;
    tx_conf.tx_free_thresh = 0; /* use PMD default value */
    tx_conf.txq_flags      = ETH_TXQ_FLAGS_NOMULTSEGS | ETH_TXQ_FLAGS_NOOFFLOADS;
    const unsigned num_tx_desc = system_params["IO_TXDESC_PER_HWTXQ"];

    struct rte_eth_fc_conf fc_conf;
    memset(&fc_conf, 0, sizeof(fc_conf));
    fc_conf.autoneg = 0;
    fc_conf.send_xon = 0;
    fc_conf.pause_time = 0xffff;
    fc_conf.mode = RTE_FC_NONE;

    /* Initialize per-node information. */
    for (unsigned node_idx = 0; node_idx < num_nodes; node_idx ++) {
        memset(&node_ports[node_idx], 0, sizeof(node_ports[0]));
    }

    struct rte_mempool* rx_mempools[NBA_MAX_PORTS][NBA_MAX_QUEUES_PER_PORT] = {{0,}};  // for debugging
    struct rte_mempool* newpkt_mempools[NBA_MAX_PORTS][NBA_MAX_QUEUES_PER_PORT] = {{0,}};
    struct rte_mempool* req_mempools[NBA_MAX_PORTS][NBA_MAX_QUEUES_PER_PORT] = {{0,}};
    /* Initialize NIC devices (rxq, txq). */
    for (port_idx = 0; port_idx < num_ports; port_idx++) {
        char dev_addr_buf[64], dev_filename[PATH_MAX], temp_buf[64];
        struct ether_addr macaddr;
        struct rte_eth_dev_info dev_info;
        struct rte_eth_link link_info;
        FILE *fp;

        if (!emulate_io) {

            rte_eth_dev_info_get(port_idx, &dev_info);

            /* Check the available RX/TX queues. */
            if (num_rxq_per_port > dev_info.max_rx_queues)
                rte_exit(EXIT_FAILURE, "port (%u, %s) does not support request number of rxq (%u).\n",
                         port_idx, dev_info.driver_name, num_rxq_per_port);
            if (num_txq_per_port > dev_info.max_tx_queues)
                rte_exit(EXIT_FAILURE, "port (%u, %s) does not support request number of txq (%u).\n",
                         port_idx, dev_info.driver_name, num_txq_per_port);

            rte_eth_link_get(port_idx, &link_info);
            RTE_LOG(INFO, MAIN, "port %u -- link running at %s %s, %s\n", port_idx,
                    (link_info.link_speed == ETH_LINK_SPEED_10000) ? "10G" : "lower than 10G",
                    (link_info.link_duplex == ETH_LINK_FULL_DUPLEX) ? "full-duplex" : "half-duplex",
                    (link_info.link_status == 1) ? "UP" : "DOWN");

            assert(0 == rte_eth_dev_configure(port_idx, num_rxq_per_port, num_txq_per_port, &port_conf));
            rte_eth_macaddr_get(port_idx, &macaddr);

            /* Initialize memory pool, rxq, txq rings. */
            unsigned node_idx = dev_info.pci_dev->numa_node;
            if(numa_disabled)
            	node_idx = 0;
            unsigned port_per_node = node_ports[node_idx].num_rx_ports;
            node_ports[node_idx].rx_ports[port_per_node].port_idx = port_idx;
            ether_addr_copy(&macaddr, &node_ports[node_idx].rx_ports[port_per_node].addr);
            node_ports[node_idx].num_rx_ports ++;
            for (ring_idx = 0; ring_idx < num_txq_per_port; ring_idx++) {
                ret = rte_eth_tx_queue_setup(port_idx, ring_idx, num_tx_desc, node_idx, &tx_conf);
                if (ret < 0)
                    rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup: err=%d, port=%d, qidx=%d\n",
                             ret, port_idx, ring_idx);
            }
            for (ring_idx = 0; ring_idx < num_rxq_per_port; ring_idx++) {
                struct mbuf_pool_ctor_arg mbp_ctor_arg;
                struct mbuf_ctor_arg mb_ctor_arg;
                char mempool_name[RTE_MEMPOOL_NAMESIZE];
                snprintf(mempool_name, RTE_MEMPOOL_NAMESIZE,
                         "pktbuf_n%u_d%u_r%u", node_idx, port_idx, ring_idx);
                struct rte_mempool *mp;

                mbp_ctor_arg.seg_buf_size = (RTE_PKTMBUF_HEADROOM + NBA_MAX_PACKET_SIZE);
                mb_ctor_arg.seg_buf_offset = ALIGN(sizeof(struct rte_mbuf), CACHE_LINE_SIZE);
                mb_ctor_arg.seg_buf_size = mbp_ctor_arg.seg_buf_size;
                uint32_t mb_size = mb_ctor_arg.seg_buf_offset + mb_ctor_arg.seg_buf_size;
                const unsigned num_mp_cache = 250;
                const unsigned num_mbufs = num_rx_desc + num_tx_desc
                                           + (num_lcores * num_mp_cache)
                                           + system_params["IO_BATCH_SIZE"];

                mp = rte_mempool_create(mempool_name, num_mbufs, mb_size, num_mp_cache,
                                        sizeof(struct rte_pktmbuf_pool_private),
                                        pktmbuf_pool_ctor, &mbp_ctor_arg,
                                        pktmbuf_obj_ctor, &mb_ctor_arg,
                                        node_idx, 0);
                if (mp == NULL)
                    rte_exit(EXIT_FAILURE, "cannot allocate memory pool for rxq %u:%u@%u.\n",
                                   port_idx, ring_idx, node_idx);
                ret = rte_eth_rx_queue_setup(port_idx, ring_idx, num_rx_desc,
                                             node_idx, &rx_conf, mp);
                if (ret < 0)
                    rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup: err=%d, port=%d, qidx=%d\n",
                             ret, port_idx, ring_idx);
                rx_mempools[port_idx][ring_idx] = mp;

                snprintf(mempool_name, RTE_MEMPOOL_NAMESIZE,
                         "newbuf_n%u_d%u_r%u", node_idx, port_idx, ring_idx);
                mp = rte_mempool_create(mempool_name, num_mbufs, mb_size, num_mp_cache,
                                        sizeof(struct rte_pktmbuf_pool_private),
                                        pktmbuf_pool_ctor, &mbp_ctor_arg,
                                        pktmbuf_obj_ctor, &mb_ctor_arg,
                                        node_idx, 0);
                if (mp == NULL)
                    rte_exit(EXIT_FAILURE, "cannot allocate new pool for rxq %u:%u@%u.\n",
                            port_idx, ring_idx, node_idx);
                newpkt_mempools[port_idx][ring_idx] = mp;

                snprintf(mempool_name, RTE_MEMPOOL_NAMESIZE,
                         "reqbuf_n%u_d%u_r%u", node_idx, port_idx, ring_idx);
                mp = rte_mempool_create(mempool_name, NUM_MBUF, mb_size, 30,
                                        sizeof(struct new_packet),
                                        0, NULL,
                                        0, NULL,
                                        node_idx, 0);
                if (mp == NULL)
                    rte_exit(EXIT_FAILURE, "cannot allocate new pool for rxq %u:%u@%u.\n",
                             port_idx, ring_idx, node_idx);
                req_mempools[port_idx][ring_idx] = mp;
            }

            /* Start RX/TX processing in the NIC! */
            assert(0 == rte_eth_dev_start(port_idx));
            //rte_eth_dev_flow_ctrl_set(port_idx, &fc_conf);
            rte_eth_promiscuous_enable(port_idx);

        } // endif emulate_io

        RTE_LOG(INFO, MAIN, "port %d is enabled.\n", port_idx);
    }

    /* Prepare queues. */
    struct rte_ring **queues;
    void **queue_privs;
    struct ev_async **qwatchers;
    {
        queues = new struct rte_ring*[queue_confs.size()];
        qwatchers = new struct ev_async*[queue_confs.size()];
        queue_privs = new void*[queue_confs.size()];

        unsigned qidx = 0;
        for (struct queue_conf &conf : queue_confs) {
            char ring_name[RTE_RING_NAMESIZE];
            unsigned queue_length = 0;
            switch (conf.template_) {
            case SWRXQ:
                queue_length = system_params["COMP_RXQ_LENGTH"];
                break;
            case TASKINQ:
                queue_length = system_params["COPROC_INPUTQ_LENGTH"];
                break;
            case TASKOUTQ:
                queue_length = system_params["COPROC_COMPLETIONQ_LENGTH"];
                break;
            default:
                assert(0); // invalid template value?
            }
            snprintf(ring_name, RTE_RING_NAMESIZE,
                     "queue%u@%u/%u", qidx, conf.node_id, conf.template_);
            queues[qidx]  = rte_ring_create(ring_name, queue_length, conf.node_id,
                                            (conf.mp_enq ? 0 : RING_F_SP_ENQ) | (conf.mc_deq ? 0 : RING_F_SC_DEQ));
            assert(queues[qidx] != NULL);
            assert(0 == rte_ring_set_water_mark(queues[qidx], queue_length - 16));
            qwatchers[qidx] = (struct ev_async *) rte_malloc_socket("ev_async", sizeof(struct ev_async),
                                                                    64, conf.node_id);
            assert(qwatchers[qidx] != NULL);
            ev_async_init(qwatchers[qidx], invalid_cb);  /* Callbacks are set by individual threads. */
            qidx ++;
        }
    }

    num_io_threads = io_thread_confs.size();
    num_comp_threads = comp_thread_confs.size();
    num_coproc_threads = coproc_thread_confs.size();

    RTE_LOG(INFO, MAIN, "%u io threads, %u comp threads, %u coproc threads (in total)\n",
                num_io_threads, num_comp_threads, num_coproc_threads);
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    signal(SIGUSR1, SIG_IGN);

    /* Some sanity checks... */
    if (emulate_io) {
        long expected_inflight_batches = NUM_MBUF / num_io_threads / system_params["COMP_PPDEPTH"];
        RTE_LOG(DEBUG, MAIN, "coproc_ppdepth = %ld, max.# in-flight batches per IO thread = %ld\n",
                             system_params["COPROC_PPDEPTH"], expected_inflight_batches);
        //if (system_params["COPROC_PPDEPTH"] > expected_inflight_batches) {
        //    rte_exit(EXIT_FAILURE, "COPROC_PPDEPTH (%ld) larger than expected max.# in-flight batches in each IO thread (%ld)"
        //                           "may block the offloading process indefinitely in the emulation mode.\n",
        //                           system_params["COPROC_PPDEPTH"], expected_inflight_batches);
        //}
    }

    /* Now we need to spawn IO, computation, corprocessor threads.
     * They have interdependencies of element graphs and device initialization steps as follows.
     * All these steps MUST be serialized although the same type of threads may run in parallel.
     *
     *  [main]                        [comp]                [coproc]                              [io]
     *  init DPDK                     .                     .                                     .
     *  read configuration            .                     .                                     .
     *  init NICs and HW queues       .                     .                                     .
     *  create SW queues              .                     .                                     .
     *  register signal handlers      .                     .                                     .
     *  (we are here now)             .                     .                                     .
     *  init coproc_ctx fields        .                     .                                     .
     *  spawn coproc threads--------------------------------+                                     .
     *  :                             .                     init event loop                       .
     *  :                             .                     init ComputeDevice & ComputeContexts  .
     *  create NodeLocalStorage       .                     :                                     .
     *  init comp_ctx fields          .                     :                                     .
     *  build elemgraph               .                     :                                     .
     *  init elemgraph (global)       .                     :                                     .
     *  init elemgraph (per-node)     .                     :                                     .
     *  :                             .                     init elemgraph (offloadables)         .
     *  init elemgraph (per-thread)   .                     :                                     .
     *  spawn comp threads------------+                     :                                     .
     *  :                             alloc job/task/batch mempool                                .
     *  :                             init event loop       :                                     .
     *  :                             :                     start event loop                      .
     *  :                             start event loop      :                                     .
     *  init io_ctx fields            :                     :                                     .
     *  spawn io threads--------------------------------------------------------------------------+
     *  :                             :                     :                                     init event loop
     *  :                             :                     :                                     start event loop
     *  (parallel execution and data-path starts!)
     */

    /* Spawn the coprocessor handler threads. */
    num_coprocessors = coproc_thread_confs.size();
    coprocessor_threads = new struct spawned_thread[num_coprocessors];
    {
        /* per-node data structures */
        unsigned per_node_counts[NBA_MAX_NODES] = {0,};

        /* per-thread initialization */
        for (i = 0; i < num_coprocessors; i++) {
            struct coproc_thread_conf &conf = coproc_thread_confs[i];
            unsigned node_id = numa_node_of_cpu(conf.core_id);
            if(numa_disabled)
            	node_id = 0;
            struct coproc_thread_context *ctx = (struct coproc_thread_context *) rte_malloc_socket(
                    "coproc_thread_conf", sizeof(*ctx), 64, node_id);

            ctx->loc.node_id = node_id;
            ctx->loc.local_thread_idx = per_node_counts[node_id] ++;
            ctx->loc.core_id = conf.core_id;

            ctx->terminate_watcher = (struct ev_async *) rte_malloc_socket(NULL, sizeof(struct ev_async), 64, node_id);
            ev_async_init(ctx->terminate_watcher, NULL);
            coprocessor_threads[i].terminate_watcher = ctx->terminate_watcher;
            coprocessor_threads[i].coproc_ctx = ctx;
            ctx->thread_init_done_barrier = new CountedBarrier(1);
            ctx->offloadable_init_barrier = new CountedBarrier(1);
            ctx->offloadable_init_done_barrier = new CountedBarrier(1);
            ctx->loopstart_barrier = new CountedBarrier(1);
            ctx->comp_ctx_to_init_offloadable = NULL;
            ctx->task_input_queue_size = system_params["COPROC_INPUTQ_LENGTH"];
            ctx->device_id = conf.device_id;
            /*
             * Assume each node has identical number of computation threads.
             * Just count node 0's computation threads.
             * (and this should work for the debugging configuration
             * which has only one computation thread in node 0.)
             */
            unsigned cnt = 0;
            for (unsigned j = 0; j < comp_thread_confs.size(); j++) {
                if (numa_node_of_cpu(comp_thread_confs[j].core_id) == 0)
                    cnt ++;
            }
            ctx->num_comp_threads_per_node = cnt;

            /* The constructor is called in coproc threads. */
            char ring_name[RTE_RING_NAMESIZE];
            snprintf(ring_name, RTE_RING_NAMESIZE, "coproc.input.%u.%u@%u",
                     ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
            ctx->task_input_queue   = queues[conf.taskinq_idx];
            ctx->task_input_watcher = qwatchers[conf.taskinq_idx];
            ctx->task_done_queue   = nullptr;
            ctx->task_done_watcher = nullptr;

            /* WARNING: subclasses are usually LARGER than their base
             * classes and malloc should use the subclass' size! */
            // TODO: (generalization) apply factory pattern for arbitrary device.
            ctx->device = NULL;
            if (dummy_device) {
                ctx->device = (ComputeDevice *) rte_malloc_socket(NULL, sizeof(DummyComputeDevice),
                                          64, ctx->loc.node_id);
            } else {
                #ifdef USE_CUDA
                ctx->device = (ComputeDevice *) rte_malloc_socket(NULL, sizeof(CUDAComputeDevice),
                                          64, ctx->loc.node_id);
                #endif
                #ifdef USE_PHI
                ctx->device = (ComputeDevice *) rte_malloc_socket(NULL, sizeof(PhiComputeDevice),
                                          64, ctx->loc.node_id);
                #endif
            }
            assert(ctx->device != NULL);

            queue_privs[conf.taskinq_idx] = ctx;

            threading::bind_cpu(ctx->loc.core_id); /* To ensure the thread is spawned in the node. */
            pthread_yield();
            assert(0 == pthread_create(&coprocessor_threads[i].tid, NULL, nba::coproc_loop, ctx));

            /* Initialize one-by-one. */
            ctx->thread_init_done_barrier->wait();
        }
    }
    RTE_LOG(INFO, MAIN, "spawned coproc threads.\n");

    /* Prepare spawning of the computation threads. */
    bool ready_flag = false;
    CondVar ready_cond;
    computation_threads = new struct spawned_thread[num_comp_threads];
    CountedBarrier *comp_init_barrier = new CountedBarrier(num_comp_threads);
    Lock *elemgraph_lock = new Lock();

    vector<comp_thread_context *> comp_thread_ctxs = vector<comp_thread_context*>();
    {
        /* per-node data structures */
        NodeLocalStorage *nls[NBA_MAX_NODES];
        unsigned per_node_counts[NBA_MAX_NODES];
        for (unsigned j = 0; j < NBA_MAX_NODES; j++) {
            nls[j] = NULL;
            per_node_counts[j] = 0;
        }

        /* per-thread initialization */
        unsigned i = 0;
        for (auto it = comp_thread_confs.begin(); it != comp_thread_confs.end(); it ++) {
            struct comp_thread_conf &conf = *it;
            unsigned node_id = numa_node_of_cpu(conf.core_id);

            if (nls[node_id] == NULL) {
                nls[node_id] = (NodeLocalStorage *) rte_malloc_socket(NULL, sizeof(NodeLocalStorage), 64, node_id);
                new (nls[node_id]) NodeLocalStorage(node_id);
            }

            comp_thread_context *ctx = (comp_thread_context *) rte_malloc_socket(
                    "comp_thread_conf", sizeof(*ctx), 64, node_id);
            new (ctx) comp_thread_context();

            ctx->loc.core_id = conf.core_id;
            ctx->loc.local_thread_idx = per_node_counts[node_id]++;
            ctx->loc.node_id = node_id;

            ctx->terminate_watcher = (struct ev_async *) rte_malloc_socket(NULL, sizeof(struct ev_async), 64, node_id);
            ev_async_init(ctx->terminate_watcher, NULL);
            computation_threads[i].terminate_watcher = ctx->terminate_watcher;
            computation_threads[i].comp_ctx = ctx;
            ctx->thread_init_barrier = comp_init_barrier;

            ctx->io_ctx = NULL;
            ctx->coproc_ctx = NULL;
            ctx->ready_flag = &ready_flag;
            ctx->ready_cond = &ready_cond;
            ctx->elemgraph_lock = elemgraph_lock;
            ctx->node_local_storage = nls[node_id];
            ctx->elem_graph = (ElementGraph *) rte_malloc_socket(NULL, sizeof(ElementGraph), 64, node_id);
            new (ctx->elem_graph) ElementGraph(ctx);
            ctx->inspector = NULL;

            ctx->num_combatch_size = system_params["COMP_BATCH_SIZE"];
            ctx->rx_queue_size = system_params["COMP_RXQ_LENGTH"];
            ctx->rx_wakeup_threshold = system_params["COMP_RXQ_THRES"];
            ctx->num_comp_ppdepth = system_params["COMP_PPDEPTH"];
            ctx->num_coproc_ppdepth = system_params["COPROC_PPDEPTH"];
            ctx->num_batchpool_size = system_params["BATCHPOOL_SIZE"];
            ctx->num_taskpool_size = system_params["TASKPOOL_SIZE"];
            ctx->task_completion_queue_size = system_params["COPROC_COMPLETIONQ_LENGTH"];
            ctx->num_tx_ports = num_ports;
            ctx->num_nodes = num_nodes;

            // TODO: extend to multiple devices
            ctx->named_offload_devices = (unordered_map<string, ComputeDevice *> *)
                    rte_malloc_socket(NULL, sizeof(unordered_map<string, ComputeDevice *>), 64, node_id);
            new (ctx->named_offload_devices) unordered_map<string, ComputeDevice *>();
            ctx->offload_devices = (vector<ComputeDevice *> *)
                    rte_malloc_socket(NULL, sizeof(vector<ComputeDevice *>), 64, node_id);
            new (ctx->offload_devices) vector<ComputeDevice *>();
            if (num_coproc_threads > 0) {
                struct coproc_thread_context *coproc_ctx = (coproc_thread_context *) queue_privs[conf.taskinq_idx];
                assert(coproc_ctx != NULL);
                ComputeDevice *device = coproc_ctx->device;
                device->input_watcher = qwatchers[conf.taskinq_idx];
                assert(coproc_ctx->task_input_watcher == device->input_watcher);
                #ifdef USE_CUDA
                ctx->named_offload_devices->insert(pair<string, ComputeDevice *>("cuda", device));
                #endif
                #ifdef USE_PHI
                ctx->named_offload_devices->insert(pair<string, ComputeDevice *>("phi", device));
                #endif
                ctx->offload_devices->push_back(device);
                ctx->offload_input_queues[0] = queues[conf.taskinq_idx];
                ctx->task_completion_queue = queues[conf.taskoutq_idx];
                ctx->task_completion_watcher = qwatchers[conf.taskoutq_idx];
                ctx->coproc_ctx = coproc_ctx;
                new (&ctx->cctx_list) FixedRing<ComputeContext *, nullptr>(2 * NBA_MAX_COPROCESSOR_TYPES, ctx->loc.node_id);
                for (unsigned k = 0, k_max = system_params["COPROC_CTX_PER_COMPTHREAD"]; k < k_max; k++) {
                    ComputeContext *cctx = nullptr;
                    cctx = device->get_available_context();
                    assert(cctx != nullptr);
                    assert(cctx->state == ComputeContext::READY);
                    ctx->cctx_list.push_back(cctx);
                }
            } else {
                new (&ctx->cctx_list) FixedRing<ComputeContext *, nullptr>(2 * NBA_MAX_COPROCESSOR_TYPES, ctx->loc.node_id);
                assert(ctx->cctx_list.empty());
                ctx->task_completion_queue = NULL;
                ctx->task_completion_watcher = NULL;
                ctx->coproc_ctx = NULL;
            }

            ctx->rx_queue = queues[conf.swrxq_idx];
            ctx->rx_watcher = qwatchers[conf.swrxq_idx];
            queue_privs[conf.swrxq_idx] = (void *) ctx;

            ctx->build_element_graph(pipeline_config);
            comp_thread_ctxs.push_back(ctx);
            i++;
        }
    }

    /* Initialze elements for this system. (once per elements) */
    {
        comp_thread_context* ctx = (comp_thread_context *) *(comp_thread_ctxs.begin());
        threading::bind_cpu(ctx->loc.core_id);
        ctx->initialize_graph_global();
    }

    /* Initialize elements for each NUMA node. */
    bool node_initialized[NBA_MAX_NODES];
    for (unsigned node_id = 0; node_id < num_nodes; ++node_id) {
        node_initialized[node_id] = false;
    }
    for (unsigned node_id = 0; node_id < num_nodes; ++node_id) {
        for (auto it = comp_thread_ctxs.begin(); it != comp_thread_ctxs.end(); ++it) {
            comp_thread_context *ctx = (comp_thread_context *) *it;
            if (ctx->loc.node_id == node_id && !node_initialized[node_id]) {
                threading::bind_cpu(ctx->loc.core_id);
                ctx->initialize_graph_per_node();
                node_initialized[node_id] = true;
            }
        }
    }

    /* Initialize offloadable elements in coprocessor threads. */
    if (num_coproc_threads > 0) {
        for (unsigned node_id = 0; node_id < num_nodes; ++node_id) {
            node_initialized[node_id] = false;
        }
        for (unsigned node_id = 0; node_id < num_nodes; ++node_id) {
            // TODO: generalize mapping of core and coprocessors
            for (auto it = comp_thread_ctxs.begin(); it != comp_thread_ctxs.end(); ++it) {
                comp_thread_context *ctx= (comp_thread_context *) *it;
                if (ctx->loc.node_id == node_id && !node_initialized[node_id]) {
                    RTE_LOG(DEBUG, MAIN, "initializing offloadables in coproc-thread@%u(%u) with comp-thread@%u\n",
                            coprocessor_threads[node_id].coproc_ctx->loc.core_id,
                            node_id, ctx->loc.core_id);
                    coprocessor_threads[node_id].coproc_ctx->comp_ctx_to_init_offloadable = ctx;
                    coprocessor_threads[node_id].coproc_ctx->offloadable_init_barrier->proceed();
                    coprocessor_threads[node_id].coproc_ctx->offloadable_init_done_barrier->wait();
                    node_initialized[node_id] = true;
                }
            }
        }
    }

    /* Initialize elements for each computation thread. */
    for (auto it = comp_thread_ctxs.begin(); it != comp_thread_ctxs.end(); ++it) {
        comp_thread_context *ctx = (comp_thread_context *) *it;
        threading::bind_cpu(ctx->loc.core_id);
        ctx->initialize_graph_per_thread();
    }

    /* Let the coprocessor threads to run its loop as we initialized
     * all necessary stuffs including computation threads. */
    for (unsigned i = 0; i < num_coprocessors; ++i) {
        coprocessor_threads[i].coproc_ctx->loopstart_barrier->proceed();
    }

    /* Spawn the IO threads. */
    io_threads = new struct spawned_thread[num_io_threads];
    {
        /* per-node data structures */
        struct io_node_stat **node_stats = new struct io_node_stat*[num_nodes];
        bool **init_done_flags = new bool*[num_nodes];
        CondVar **init_conds = new CondVar*[num_nodes];
        struct ev_async **node_stat_watchers = new ev_async*[num_nodes];
        rte_atomic16_t **node_master_flags = new rte_atomic16_t*[num_nodes];
        io_thread_context **node_master_ctxs = new io_thread_context*[num_nodes];

        for (unsigned node_id = 0; node_id < num_nodes; node_id ++) {
            node_stats[node_id] = (struct io_node_stat *) rte_malloc_socket("io_node_stat", sizeof(struct io_node_stat),
                                                                            CACHE_LINE_SIZE, node_id);
            node_stats[node_id]->node_id = node_id;
            node_stats[node_id]->num_ports = num_ports;
            node_stats[node_id]->last_time = 0;
            for (j = 0; j < node_stats[node_id]->num_ports; j++) {
                node_stats[node_id]->port_stats[j].num_recv_pkts = RTE_ATOMIC64_INIT(0);
                node_stats[node_id]->port_stats[j].num_sent_pkts = RTE_ATOMIC64_INIT(0);
                node_stats[node_id]->port_stats[j].num_sw_drop_pkts = RTE_ATOMIC64_INIT(0);
                node_stats[node_id]->port_stats[j].num_rx_drop_pkts = RTE_ATOMIC64_INIT(0);
                node_stats[node_id]->port_stats[j].num_tx_drop_pkts = RTE_ATOMIC64_INIT(0);
                node_stats[node_id]->port_stats[j].num_invalid_pkts = RTE_ATOMIC64_INIT(0);
            }
            memset(&node_stats[node_id]->last_total, 0, sizeof(struct io_thread_stat));
            unsigned num_io_threads_in_node = 0;
            for (auto it = io_thread_confs.begin(); it != io_thread_confs.end(); it++) {
                struct io_thread_conf &conf = *it;
                if (numa_node_of_cpu(conf.core_id) == (signed) node_id)
                    num_io_threads_in_node ++;
            }
            node_stats[node_id]->num_threads = num_io_threads_in_node;

            node_stat_watchers[node_id] = (struct ev_async *) rte_malloc_socket(nullptr, sizeof(struct ev_async),
                                                                                CACHE_LINE_SIZE, node_id);

            node_master_flags[node_id] = (rte_atomic16_t *) rte_malloc_socket(nullptr, sizeof(rte_atomic16_t),
                                                                              CACHE_LINE_SIZE, node_id);

            node_master_ctxs[node_id] = nullptr;

            init_done_flags[node_id] = (bool *) rte_malloc_socket("io_ctx.initflag", sizeof(bool),
                                                                  CACHE_LINE_SIZE, node_id);
            *init_done_flags[node_id] = false;
            init_conds[node_id] = (CondVar *) rte_malloc_socket("io_ctx.condvar", sizeof(CondVar),
                                                                CACHE_LINE_SIZE, node_id);
            new (init_conds[node_id]) CondVar();
        }
        unsigned per_node_counts[NBA_MAX_NODES] = {0,};

        /* per-thread initialization */
        srand(time(0));
        unsigned i = 0;
        for (auto it = io_thread_confs.begin(); it != io_thread_confs.end(); it++) {
            struct io_thread_conf &conf = *it;
            unsigned node_id = numa_node_of_cpu(conf.core_id);

            struct io_thread_context *ctx = (struct io_thread_context *) rte_malloc_socket(
                    "io_thread_conf", sizeof(*ctx), CACHE_LINE_SIZE, node_id);
            char ring_name[RTE_RING_NAMESIZE];

            ctx->loc.core_id    = conf.core_id;
            ctx->loc.local_thread_idx = per_node_counts[node_id] ++;
            ctx->loc.node_id    = node_id;
            ctx->loc.global_thread_idx = i;

            if (ctx->loc.local_thread_idx == 0)
                node_master_ctxs[node_id] = ctx;
            ctx->node_master_ctx = node_master_ctxs[node_id];
            assert(ctx->node_master_ctx != nullptr);

            ctx->comp_ctx = NULL;
            ctx->block = (CondVar *) rte_malloc_socket(NULL, sizeof(CondVar), CACHE_LINE_SIZE, node_id);
            new (ctx->block) CondVar();
            ctx->is_block = false;
            ctx->terminate_watcher = (struct ev_async *) rte_malloc_socket(NULL, sizeof(struct ev_async),
                                                                           CACHE_LINE_SIZE, node_id);
            ev_async_init(ctx->terminate_watcher, NULL);
            ctx->io_lock = (Lock *) rte_malloc_socket(NULL, sizeof(Lock), CACHE_LINE_SIZE, node_id);
            new (ctx->io_lock) Lock();
            ctx->init_cond = init_conds[node_id];
            ctx->init_done = init_done_flags[node_id];
            ctx->node_stat = node_stats[node_id];
            ctx->node_stat_watcher = node_stat_watchers[node_id];
            ctx->node_master_flag = node_master_flags[node_id];
            ctx->random_seed = rand();

            ctx->num_io_threads = num_io_threads;
            ctx->num_iobatch_size = system_params["IO_BATCH_SIZE"];
            ctx->mode = conf.mode;
            ctx->LB_THRUPUT_WINDOW_SIZE = (1 << 16);

            /* drop_queue, tx_queue, prepacket_queue are
             * one-to-one mapped to IO threads.
             * Multiple computation threads may become the
             * producers of them owned by a single IO thread.
             */
            snprintf(ring_name, RTE_RING_NAMESIZE, "dropq.%u:%u@%u",
                     ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
            ctx->drop_queue = rte_ring_create(ring_name, 8 * NBA_MAX_COMPBATCH_SIZE,
                                              node_id, RING_F_SC_DEQ);
            assert(NULL != ctx->drop_queue);

            ctx->num_tx_ports = num_ports;
            for (k = 0; k < num_ports; k++) {
                snprintf(ring_name, RTE_RING_NAMESIZE, "txq%u.%u:%u@%u",
                         k, ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
                ctx->tx_queues[k] = rte_ring_create(ring_name, 8 * NBA_MAX_COMPBATCH_SIZE,
                                    node_id, RING_F_SC_DEQ);
                assert(NULL != ctx->tx_queues[k]);
                assert(0 == rte_ring_set_water_mark(ctx->tx_queues[k], (8 * NBA_MAX_COMPBATCH_SIZE) - 16));
            }

            snprintf(ring_name, RTE_RING_NAMESIZE, "reqring.%u:%u@%u",
                     ctx->loc.node_id, ctx->loc.local_thread_idx, ctx->loc.core_id);
            ctx->new_packet_request_ring = rte_ring_create(ring_name, NUM_MBUF, node_id, RING_F_SC_DEQ);
            assert(NULL != ctx->new_packet_request_ring);

            ctx->num_hw_rx_queues = conf.attached_rxqs.size();

            unsigned k = 0;
            char mempool_name[RTE_MEMPOOL_NAMESIZE];
            for (auto itq = conf.attached_rxqs.begin(); itq != conf.attached_rxqs.end(); itq ++) {
                struct hwrxq rxq = *itq;
                ctx->rx_hwrings[k] = rxq;
                ctx->rx_pools[k] = rx_mempools[itq->ifindex][itq->qidx];
                k++;
            }
            if (emulate_io) {
                uint32_t mb_size = ALIGN(sizeof(struct rte_mbuf), CACHE_LINE_SIZE)
                                   + (RTE_PKTMBUF_HEADROOM + NBA_MAX_PACKET_SIZE);
                snprintf(mempool_name, RTE_MEMPOOL_NAMESIZE,
                         "emul_pktbuf_%u:%u", ctx->loc.node_id, ctx->loc.core_id);
                ctx->emul_rx_packet_pool = rte_mempool_create(mempool_name,
                        ctx->num_iobatch_size * conf.attached_rxqs.size() * system_params["COPROC_PPDEPTH"],
                        mb_size, CACHE_LINE_SIZE, sizeof(rte_pktmbuf_pool_private),
                        rte_pktmbuf_pool_init, NULL, rte_pktmbuf_init, NULL,
                        ctx->loc.node_id, 0);
                assert(ctx->emul_rx_packet_pool != NULL);
            }
            //snprintf(mempool_name, RTE_MEMPOOL_NAMESIZE,
            //         "newbuf_%u:%u", ctx->loc.node_id, ctx->loc.core_id);
            //ctx->new_packet_pool = rte_mempool_create(mempool_name, ctx->num_iobatch_size * conf.attached_rxqs.size(),
            //                                          MBUF_SIZE, CACHE_LINE_SIZE,
            //                                          sizeof(rte_pktmbuf_pool_private),
            //                                          rte_pktmbuf_pool_init, NULL,
            //                                          rte_pktmbuf_init, NULL,
            //                                          ctx->loc.node_id, 0);
            //assert(ctx->new_packet_pool != NULL);
            //snprintf(mempool_name, RTE_MEMPOOL_NAMESIZE,
            //         "reqbuf_%u:%u", ctx->loc.node_id, ctx->loc.core_id);
            //ctx->new_packet_request_pool = rte_mempool_create(mempool_name, ctx->num_iobatch_size * conf.attached_rxqs.size(),
            //                                                  MBUF_SIZE, CACHE_LINE_SIZE,
            //                                                  sizeof(rte_pktmbuf_pool_private),
            //                                                  rte_pktmbuf_pool_init, NULL,
            //                                                  rte_pktmbuf_init, NULL,
            //                                                  ctx->loc.node_id, 0);
            //assert(ctx->new_packet_request_pool != NULL);
            ctx->rx_queue   = queues[conf.swrxq_idx];
            ctx->rx_watcher = qwatchers[conf.swrxq_idx];
            if (emulate_io) {
                ctx->emul_packet_size = emulated_packet_size;
                ctx->emul_ip_version = emulated_ip_version;
            }

            assert(i < num_io_threads);
            io_threads[i].terminate_watcher = ctx->terminate_watcher;
            io_threads[i].io_ctx = ctx;

            comp_thread_context *comp_ctx = (comp_thread_context *) queue_privs[conf.swrxq_idx];
            assert(comp_ctx != NULL);
            RTE_LOG(DEBUG, MAIN, "   mapping io thread %u and comp thread @%u\n",
                    ctx->loc.core_id, comp_ctx->loc.core_id);
            comp_ctx->io_ctx = ctx;
            ctx->comp_ctx = comp_ctx;
            i++;
        }
    }

    /* Notify computation threads to be ready. */
    ready_cond.lock();
    ready_flag = true;
    ready_cond.signal_all();
    ready_cond.unlock();

    struct thread_collection col;
    col.num_io_threads = num_io_threads;
    col.io_threads     = io_threads;
    RTE_LOG(INFO, MAIN, "spawned io threads.\n");
    RTE_LOG(INFO, MAIN, "running...\n");

    /* Since we set CALL_MASTER, this function blocks until the master
     * finishes. (master = io_loop[0:0@0]) */
    rte_eal_mp_remote_launch(nba::thread_wrapper, &col, CALL_MASTER);

    /* Wait until the spawned threads are finished. */
    _exit_cond.lock();
    while (!_terminated) {
        _exit_cond.wait();
    }
    _exit_cond.unlock();

    RTE_LOG(NOTICE, MAIN, "terminated.\n");
    return 0;
}

static void handle_signal(int) {
    unsigned i;

    if (threading::is_thread_equal(main_thread_id, threading::self())) {
        RTE_LOG(NOTICE, MAIN, "terminating...\n");

        for (i = 0; i < num_coproc_threads; i++)
            ev_async_send(coprocessor_threads[i].coproc_ctx->loop, coprocessor_threads[i].terminate_watcher);

        for (i = 0; i < num_io_threads; i++)
            ev_async_send(io_threads[i].io_ctx->loop, io_threads[i].terminate_watcher);
        for (i = 0; i < num_coproc_threads; i++)
            pthread_join(coprocessor_threads[i].tid, NULL);
        rte_eal_mp_wait_lcore();

        /* Set the terminated flag. */
        _exit_cond.lock();
        _terminated = true;
        _exit_cond.signal();
        _exit_cond.unlock();
    }
}

// vim: ts=8 sts=4 sw=4 et
