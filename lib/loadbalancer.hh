#ifndef __LOADBALANCER_HH__
#define __LOADBALANCER_HH__

#include <vector>
#include "packetbatch.hh"
#include "computedevice.hh"

namespace nba {

/**
 * The System Inspector Interface.
 *
 * It is used by user-implemented load balancers to inspect system status
 * and update their internal load balancing parametes.
 */
class SystemInspector {
public:
    SystemInspector() :
            rx_batch_count(0), rx_pkt_count(0),
            tx_batch_count(0), tx_pkt_count(0),
            drop_pkt_count(0), batch_process_time(0), tx_pkt_thruput(0),
            true_process_time_cpu(0)
    {
        for (unsigned i = 0; i < NBA_MAX_COPROCESSOR_TYPES; i++) {
            dev_sent_batch_count[i] = 0;
            dev_finished_batch_count[i] = 0;
            dev_finished_task_count[i] = 0;
            avg_task_completion_sec[i] = 0;
            true_process_time_gpu[i] = 0;
        }
    }

    virtual ~SystemInspector() { }

    /* We do not use wrapper methods to write/read these values, since
     * there is no race condition as all fields are accessed
     * exclusively by a single computation thread. */
    uint64_t dev_sent_batch_count[NBA_MAX_COPROCESSOR_TYPES];
    uint64_t dev_finished_batch_count[NBA_MAX_COPROCESSOR_TYPES];
    uint64_t dev_finished_task_count[NBA_MAX_COPROCESSOR_TYPES];
    float avg_task_completion_sec[NBA_MAX_COPROCESSOR_TYPES];
    uint64_t rx_batch_count;
    uint64_t rx_pkt_count;
    uint64_t tx_batch_count;
    uint64_t tx_pkt_count;
    uint64_t drop_pkt_count;
    uint64_t batch_process_time;

    double tx_pkt_thruput;

    double true_process_time_gpu[NBA_MAX_COPROCESSOR_TYPES];
    double true_process_time_cpu;
#define CPU_HISTORY_SIZE (128)
#define GPU_HISTORY_SIZE (2048)
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
