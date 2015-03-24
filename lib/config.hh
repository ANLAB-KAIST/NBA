#ifndef __NBA_CONFIG_HH__
#define __NBA_CONFIG_HH__

#include <unordered_map>
#include <vector>
#include <string>

#define NBA_MAX_NODES       (2)
#define NBA_MAX_CORES       (64)
#define NBA_MAX_PORTS       (16)
#define NBA_MAX_QUEUES_PER_PORT (128)
#define NBA_MAX_COPROCESSORS      (2)  // max number of coprocessor devices
#define NBA_MAX_COPROCESSOR_TYPES (1)  // max number of device types
#define NBA_MAX_PACKET_SIZE     (2048)
#ifdef NBA_NO_HUGE
  #define NBA_MAX_IOBATCH_SIZE    (4u)
  #define NBA_MAX_COMPBATCH_SIZE  (4u)
#else
  #define NBA_MAX_IOBATCH_SIZE    (256u)
  #define NBA_MAX_COMPBATCH_SIZE  (256u)
#endif
#define NBA_MAX_SW_RXRING_LENGTH  (2048u)
#define NBA_MAX_COMP_PPDEPTH   (256u)
#define NBA_MAX_COPROC_PPDEPTH (32u)
#define NBA_MAX_BATCHPOOL_SIZE  (2048u)
#define NBA_MAX_ANNOTATION_SET_SIZE (7)  // maximum possible: 64
#define NBA_MAX_NODELOCALSTORAGE_ENTRIES    (16)
#define NBA_MAX_KERNEL_OVERLAP  (8)
#define NBA_MAX_DATABLOCKS (12)  // If too large (e.g., 64), batch_pool can not be allocated.
#define NBA_OQ  (true)  // Use output-queuing semantics when possible.

namespace nba {

enum io_thread_mode {
    IO_NORMAL = 0,
    IO_ECHOBACK = 1,
    IO_RR = 2,
    IO_RXONLY = 3,
    IO_EMUL = 4,
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
extern bool emulate_io;
extern int emulated_packet_size;
extern int emulated_ip_version;
extern int emulated_num_fixed_flows;
extern size_t num_emulated_ifaces;

bool load_config(const char* pyfilename);
bool check_ht_enabled();

}

#undef HAVE_PSIO

/* For microbenchmarks (see lib/io.cc) */
//#define TEST_MINIMAL_L2FWD
//#define TEST_RXONLY

/* Inserts forced sleep when there is no packets received,
 * to reduce PCIe traffic.  The performance may increase or decrease
 * depending on the system configuration.
 * (see lib/io.cc)
 */
//#define NBA_SLEEPY_IO

#endif

// vim: ts=8 sts=4 sw=4 et
