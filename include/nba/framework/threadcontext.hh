#ifndef __NBA_THREADCONTEXT_HH__
#define __NBA_THREADCONTEXT_HH__

#include <nba/core/intrinsic.hh>
#include <nba/core/queue.hh>
#include <nba/framework/config.hh>
#include <cstdint>
#include <cstdbool>
#include <string>
#include <set>
#include <vector>
#include <unordered_map>
#include <functional>
#include <pthread.h>
#include <unistd.h>
#include <ev.h>
#include <rte_config.h>
#include <rte_atomic.h>
#include <rte_ring.h>
#include <rte_ether.h>

namespace nba {

/* forward declarations */
class CondVar;
class CountedBarrier;
class Lock;
class Element;
class PacketBatch;
class DataBlock;
class SystemInspector;
class ElementGraph;
class ComputeDevice;
class ComputeContext;
class NodeLocalStorage;
class OffloadTask;
class comp_thread_context;
struct io_port_stat;

struct core_location {
    unsigned node_id;
    unsigned core_id;   /** system-global core index (used thread pinning) */
    unsigned local_thread_idx;  /** node-local thread index (used to access data structures) */
    unsigned global_thread_idx; /** global thread index (used to access data structures) */
} __cache_aligned;

struct port_info {
    unsigned port_idx;
    struct ether_addr addr;
} __cache_aligned;

struct new_packet
{
    char buf[NBA_MAX_PACKET_SIZE];
    size_t len;
    int out_port;
};

/* Thread arguments for each types of thread */

struct io_thread_context {
    struct ev_async *terminate_watcher;
    CondVar *init_cond;
    bool *init_done;
    Lock *io_lock;

    char _reserved0[64]; // to prevent false-sharing

    struct core_location loc;
    struct ev_loop *loop;
    bool loop_broken;
    unsigned num_hw_rx_queues;
    unsigned num_tx_ports;
    unsigned num_iobatch_size;
    unsigned num_io_threads;
    uint64_t last_tx_tick;
    uint64_t global_tx_cnt;
    uint64_t tx_pkt_thruput;
    uint64_t LB_THRUPUT_WINDOW_SIZE;
    int emul_packet_size;
    int emul_ip_version;
    int mode;
    struct hwrxq rx_hwrings[NBA_MAX_PORTS * NBA_MAX_QUEUES_PER_PORT];
    struct ev_timer *stat_timer;
    struct io_port_stat *port_stats;
    struct io_thread_context *node_master_ctx;
#ifdef NBA_CPU_MICROBENCH
    int papi_evset_rx;
    int papi_evset_tx;
    int papi_evset_comp;
    long long papi_ctr_rx[5];
    long long papi_ctr_tx[5];
    long long papi_ctr_comp[5];
#endif

    char _reserved1[64];

    struct rte_ring *rx_queue;
    struct ev_async *rx_watcher;
    struct port_info tx_ports[NBA_MAX_PORTS];
    comp_thread_context *comp_ctx;

    char _reserved2[64]; // to prevent false-sharing

    struct rte_ring *drop_queue;
    struct rte_ring *tx_queues[NBA_MAX_PORTS];

    struct rte_mempool* rx_pools[NBA_MAX_PORTS];
    struct rte_mempool* emul_rx_packet_pool = nullptr;
    struct rte_mempool* new_packet_pool = nullptr;
    struct rte_mempool* new_packet_request_pool = nullptr;
    struct rte_ring* new_packet_request_ring = nullptr;

    char _reserved3[64]; // to prevent false-sharing

    pid_t thread_id;
    CondVar *block;
    bool is_block;
    unsigned int random_seed;

    char _reserved4[64]; /* prevent false-sharing */

    rte_atomic16_t *node_master_flag;
    struct ev_async *node_stat_watcher;
    struct io_node_stat *node_stat;

} __cache_aligned;

class comp_thread_context {
public:
    comp_thread_context();
    virtual ~comp_thread_context();
    void stop_rx();
    void resume_rx();

    void build_element_graph(const char* config);       // builds element graph
    void initialize_graph_global();
    void initialize_graph_per_node();
    void initialize_graph_per_thread();
    void initialize_offloadables_per_node(ComputeDevice *device);
    void io_tx_new(void* data, size_t len, int out_port);
public:
    struct ev_async *terminate_watcher;
    CountedBarrier *thread_init_barrier;
    CondVar *ready_cond;
    bool *ready_flag;
    Lock *elemgraph_lock;
    NodeLocalStorage *node_local_storage;

    char _reserved1[64]; /* prevent false-sharing */

    struct ev_loop *loop;
    struct core_location loc;
    unsigned num_tx_ports;
    unsigned num_nodes;
    unsigned num_coproc_ppdepth;
    unsigned num_combatch_size;
    unsigned num_batchpool_size;
    unsigned num_taskpool_size;
    unsigned task_completion_queue_size;

    struct rte_mempool *batch_pool;
    struct rte_mempool *dbstate_pool;
    struct rte_mempool *task_pool;
    struct rte_mempool *packet_pool;
    ElementGraph *elem_graph;
    SystemInspector *inspector;
    FixedRing<ComputeContext *, nullptr> cctx_list;
    PacketBatch *input_batch;
    DataBlock *datablock_registry[NBA_MAX_DATABLOCKS];

    bool stop_task_batching;
    struct rte_ring *rx_queue;
    struct ev_async *rx_watcher;
    struct coproc_thread_context *coproc_ctx;

    char _reserved2[64]; /* prevent false-sharing */

    struct io_thread_context *io_ctx;
    std::unordered_map<std::string, ComputeDevice *> *named_offload_devices;
    std::vector<ComputeDevice*> *offload_devices;
    struct rte_ring *offload_input_queues[NBA_MAX_COPROCESSORS]; /* ptr to per-device task input queue */

    char _reserved3[64]; /* prevent false-sharing */

    struct rte_ring *task_completion_queue; /* to receive completed offload tasks */
    struct ev_async *task_completion_watcher;
    struct ev_check *check_watcher;
} __cache_aligned;

struct coproc_thread_context {
    struct ev_async *terminate_watcher;
    CountedBarrier *thread_init_done_barrier;
    CountedBarrier *offloadable_init_barrier;
    CountedBarrier *offloadable_init_done_barrier;
    CountedBarrier *loopstart_barrier;
    struct ev_async *offloadable_init_watcher;
    comp_thread_context *comp_ctx_to_init_offloadable;

    char _reserved1[64]; // to prevent false-sharing

    struct core_location loc;
    struct ev_loop *loop;
    bool loop_broken;
    unsigned device_id;
    unsigned num_comp_threads_per_node;
    unsigned task_input_queue_size;
    ComputeDevice *device;

    struct ev_async *task_d2h_watcher;
    FixedRing<OffloadTask *, nullptr> *d2h_pending_queue;
    FixedRing<OffloadTask *, nullptr> *task_done_queue;
    struct ev_async *task_done_watcher;

    char _reserved2[64]; // to prevent false-sharing

    struct rte_ring *task_input_queue;
    struct ev_async *task_input_watcher;

    char _reserved3[64]; // to prevent false-sharing

} __cache_aligned;

struct spawned_thread {
    pthread_t tid;
    struct ev_async *terminate_watcher;
    union {
        struct io_thread_context *io_ctx;
        struct comp_thread_context *comp_ctx;
        struct coproc_thread_context *coproc_ctx;
    };
} __cache_aligned;

struct thread_collection {
    struct spawned_thread *io_threads;
    unsigned num_io_threads;
} __cache_aligned;

}

#endif

// vim: ts=8 sts=4 sw=4 et
