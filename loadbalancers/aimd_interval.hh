#ifndef __LB_AIMD_INTERVAL_HH__
#define __LB_AIMD_INTERVAL_HH__

#include "../lib/loadbalancer.hh"
#include "../lib/common.hh"

namespace nshader {

// LB using AIMD interval: Just ported old nShader's dispatcher_cc.hh
class AIMDIntervalLB : public LoadBalancer
{
public:
    AIMDIntervalLB(): LoadBalancer() { }
    virtual ~AIMDIntervalLB() {
        driver_data.prev_usec = 0;
        driver_data.interval = 100; // Dummy init value. TODO: confirm this value is appropriate.
    }

    int gate_keeper(PacketBatch *batch, vector<ComputeDevice*>& devices)
    {
        /* Determine whether to send the task to CPU or GPU,
         * by looking at the (per-task-type) interval value.
         *
         * (From the beginning of an interval to the end of it) => the task is sent to CPUs.
         * (Upon each interval event)               => the task is sent to GPUs.
         */
        uint64_t cur_usec = get_usec();
	int choice = 0;	  // default: send to the GPU
        if (driver_data.last_sent + driver_data.interval < cur_usec) {
            driver_data.last_sent = cur_usec;
            choice = -1;  // send to the CPU
        }

	if (driver_data.prev_choice == 0 && choice == -1) {
//		printf("Transition to the CPU!\n");
		is_changed_to_cpu = true;
	}

	driver_data.prev_choice = choice;
	return choice;
    }

    uint64_t update_params(SystemInspector &inspector, uint64_t timestamp)
    {
        // Update reference data & interval
        uint64_t cur_usec = timestamp;
        uint64_t prev_usec = driver_data.prev_usec;
        uint64_t interval = driver_data.interval;

        /* Updating of an interval follows the return value of this function.
         * From here, "last time" indicates that the last time when the same type of
         * task is checked. */

	// Previous code to compute "idle".
        uint64_t cpu_time = get_thread_cpu_time();
        uint64_t driver_time = cpu_time - driver_data.prev_cpu_time; /* thread cpu time between current and last tuning */
        double idle = 1.0 - (((double)driver_time / get_thread_cpu_time_unit()) \
                           / ((double)(cur_usec - prev_usec) / 1e6));
        if (idle < 0.0)
            idle = 0.0;

        /* We adjust the interval like the TCP's AIMD.
         * The unit of interval is microseconds. */
        if (idle > 0.5) {
            /* If our core is idle enough, set the
             * interval as just 1 usec so that all CPU
             * cores get the work. */
            interval /= 1.1;
        } else if (idle > 0.40) {
            /* Our CPU core is somewhat busy. */
            interval /= 1.1;  /* multiplicative decrease */
        } else {
            /* Here, our CPU core is very busy! */
            // TODO: check if this code checks "the direction of previous interval change(s)"
            interval *= 3;  /* fast doubling */
        }

        driver_data.interval = interval;
        driver_data.interval += 200;    /* additive increase (always) */
        driver_data.prev_usec = cur_usec;
        driver_data.prev_idle = idle;
        driver_data.prev_driver_time = driver_time;
        driver_data.prev_cpu_time = cpu_time;
	driver_data.prev_dev_finished_batch_cnt = inspector.dev_finished_batch_count[0];
	driver_data.prev_dev_sent_batch_cnt = inspector.dev_sent_batch_count[0];
//        printf("============== LB data updated! idle:%f, interval: %lu\n", idle, driver_data.interval);

        // Timeout for next update: 1sec (= return value / 1000000u)
        return 1000000u;
    }
    
    struct reference_data
    {
        uint64_t prev_usec;         // timestamp at the time of last-tuning
        uint64_t prev_cpu_time;     // thread cpu time at the time of last-tuning
        uint64_t prev_idle;         // for recording
        uint64_t interval;          // "THE" interval
        uint64_t last_sent;         // set when a task is sent to CPU
        uint64_t prev_driver_time;  // Thread cpu time b/t last and last-last tuning
	int prev_choice;		// Previous choice of load balancer.
        uint64_t prev_dev_finished_batch_cnt;  // # of finished batch cnt in dev in last tuning.
	uint64_t prev_dev_sent_batch_cnt; 
        // uint64_t update_count;
    } __attribute__ ((aligned(64)));

    struct reference_data driver_data;
};

}

EXPORT_LOADBALANCER(AIMDIntervalLB);

#endif

// vim: ts=8 sts=4 sw=4 et
