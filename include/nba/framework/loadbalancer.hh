#ifndef __LOADBALANCER_HH__
#define __LOADBALANCER_HH__

#include <nba/framework/config.hh>
#include <nba/framework/computedevice.hh>
#include <vector>

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
            drop_pkt_count(0), batch_proc_time(0)
    {
        for (unsigned i = 0; i < NBA_MAX_COPROCESSOR_TYPES; i++) {
            dev_sent_batch_count[i] = 0;
            dev_finished_batch_count[i] = 0;
            dev_finished_task_count[i] = 0;
            avg_task_completion_sec[i] = 0;
            pkt_proc_cycles[i] = 0;
        }
    }

    virtual ~SystemInspector() { }

    void update_pkt_proc_cycles(uint64_t val, int proc_id)
    {
        pkt_proc_cycles[proc_id] = (((pkt_proc_cycles[proc_id]
                                      * (PPC_HISTORY_SIZES[proc_id] - 1))
                                     + val)
                                    / PPC_HISTORY_SIZES[proc_id]);
    }

    void update_batch_proc_time(uint64_t val)
    {
        batch_proc_time = 0.01 * val + 0.99 * batch_proc_time;
    }

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
    uint64_t batch_proc_time;
    double pkt_proc_cycles[NBA_MAX_COPROCESSOR_TYPES + 1];

    //const unsigned PPC_HISTORY_SIZES[2] = {128, 2048};
    const unsigned PPC_HISTORY_SIZES[2] = {512, 512};
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
