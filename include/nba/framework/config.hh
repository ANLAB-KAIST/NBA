#ifndef __NBA_CONFIG_HH__
#define __NBA_CONFIG_HH__

#include <unordered_map>
#include <vector>
#include <string>

#define NBA_MAX_NODES               (2)
#define NBA_MAX_CORES               (64)
#define NBA_MAX_PORTS               (16)
#define NBA_MAX_QUEUES_PER_PORT     (128)
#define NBA_MAX_COPROCESSORS        (2)     // Max number of coprocessor devices
#define NBA_MAX_COPROCESSOR_TYPES   (1)     // Max number of coprocessor types


#define NBA_BATCHING_TRADITIONAL    (0)
#define NBA_BATCHING_CONTINUOUS     (1)
#define NBA_BATCHING_BITVECTOR      (2)
#define NBA_BATCHING_LINKEDLIST     (3)

#ifndef NBA_BATCHING_SCHEME
#define NBA_BATCHING_SCHEME         NBA_BATCHING_TRADITIONAL
#endif


#define NBA_BRANCHPRED_DISABLED     (0)    // disabled
#define NBA_BRANCHPRED_ENABLED      (1)    // conditionally perform branch-pred
#define NBA_BRANCHPRED_ALWAYS       (2)    // always perform branch-pred

#ifndef NBA_BRANCHPRED_SCHEME
#define NBA_BRANCHPRED_SCHEME       NBA_BRANCHPRED_DISABLED
#endif


#define NBA_REUSE_DATABLOCKS_DISABLED  (0)
#define NBA_REUSE_DATABLOCKS_ENABLED   (1)

#ifndef NBA_REUSE_DATABLOCKS
#define NBA_REUSE_DATABLOCKS        NBA_REUSE_DATABLOCKS_ENABLED
#endif


#define NBA_MAX_PACKET_SIZE         (2048)
#ifdef NBA_NO_HUGE
  #define NBA_MAX_IO_BATCH_SIZE      (4u)
  #define NBA_MAX_COMP_BATCH_SIZE    (4u)
#else
  #if NBA_BATCHING_SCHEME == NBA_BATCHING_BITVECTOR
    #define NBA_MAX_IO_BATCH_SIZE    (64u)
    #define NBA_MAX_COMP_BATCH_SIZE  (64u)
  #else
    #define NBA_MAX_IO_BATCH_SIZE    (256u)
    #define NBA_MAX_COMP_BATCH_SIZE  (256u)
  #endif
#endif
#define NBA_MAX_COMP_PREPKTQ_LENGTH (256u)
#if defined(NBA_PMD_MLX4) || defined(NBA_PMD_MLNX_UIO)
#define NBA_MAX_IO_DESC_PER_HWRXQ      (8192)
#define NBA_MAX_IO_DESC_PER_HWTXQ      (8192)
#else
#define NBA_MAX_IO_DESC_PER_HWRXQ      (1024)
#define NBA_MAX_IO_DESC_PER_HWTXQ      (1024)
#endif

#define NBA_MAX_COPROC_PPDEPTH      (64u)
#define NBA_MAX_COPROC_INPUTQ_LENGTH       (64)
#define NBA_MAX_COPROC_COMPLETIONQ_LENGTH  (64)
#define NBA_MAX_COPROC_CTX_PER_COMPTHREAD   (1)

#define NBA_MAX_TASKPOOL_SIZE       (2048u)
#define NBA_MAX_BATCHPOOL_SIZE      (2048u)
#define NBA_MAX_IO_BASES            (1)

#define NBA_MAX_NODELOCALSTORAGE_ENTRIES    (16)
#define NBA_MAX_KERNEL_OVERLAP      (8)
#define NBA_MAX_DATABLOCKS          (12)    // If too large (e.g., 64), batch_pool can not be allocated.

#define NBA_OQ                      (true)  // Use output-queuing semantics when possible.
#undef NBA_CPU_MICROBENCH                  // Enable support for PAPI library for microbenchmarks.

#undef NBA_IPFWD_RR_NODE_LOCAL


/* If you change below, update HANDLE_ALL_PORTS macro in lib/element.hh as well!! */
#define NBA_MAX_ELEM_NEXTS (4)

namespace nba {

enum io_thread_mode {
    IO_NORMAL = 0,
    IO_ECHOBACK = 1,
    IO_RR = 2,
    IO_RXONLY = 3,
};

enum queue_template {
    SWRXQ = 0,
    TASKINQ = 1,
    TASKOUTQ = 2,
};

struct hwrxq {
    int ifindex;
    int qidx;
};

struct io_thread_conf {
    int core_id;
    std::vector<struct hwrxq> attached_rxqs;
    int mode;
    int swrxq_idx;
    void *priv;
};

struct comp_thread_conf {
    int core_id;
    int swrxq_idx;
    int taskinq_idx;
    int taskoutq_idx;
    void *priv;
};

struct coproc_thread_conf {
    int core_id;
    int device_id;
    int taskinq_idx;
    int taskoutq_idx;
    void *priv;
};

struct queue_conf {
    int node_id;
    enum queue_template template_;
    bool mp_enq;
    bool mc_deq;
    void *priv;
};

extern std::unordered_map<std::string, long> system_params;
extern std::vector<struct io_thread_conf> io_thread_confs;
extern std::vector<struct comp_thread_conf> comp_thread_confs;
extern std::vector<struct coproc_thread_conf> coproc_thread_confs;
extern std::vector<struct queue_conf> queue_confs;
/* queue_idx_map is used to find the appropriate queue instance by
 * the initialization code of io/comp/coproc threads. */
extern std::unordered_map<void*, int> queue_idx_map;
extern bool dummy_device;

bool load_config(const char* pyfilename);
int get_ht_degree(void);

}

#undef HAVE_PSIO

/* For microbenchmarks (see lib/io.cc) */
//#define TEST_MINIMAL_L2FWD
//#define TEST_RXONLY

#endif

// vim: ts=8 sts=4 sw=4 et
