#ifndef __NBA_ELEMENT_LOADBALANCEADAPTIVEGLOBAL_HH__
#define __NBA_ELEMENT_LOADBALANCEADAPTIVEGLOBAL_HH__

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include "../../lib/loadbalancer.hh"
#include "../../lib/queue.hh"

#include <rte_errno.h>
#include <rte_log.h>
#include <rte_atomic.h>

#include <vector>
#include <string>
#include <random>

#define _LB_PPC_GLOBAL_MY_CPU_TIME (1000)
#define _LB_PPC_GLOBAL_MY_CPU_DELTA (10)

namespace nba {

class LoadBalanceAdaptiveGlobal : public SchedulableElement, PerBatchElement {
public:
    LoadBalanceAdaptiveGlobal() : SchedulableElement(), PerBatchElement()
    { }

    virtual ~LoadBalanceAdaptiveGlobal()
    { }

    const char *class_name() const { return "LoadBalanceAdaptiveGlobal"; }
    const char *port_count() const { return "1/1"; }
    int get_type() const { return SchedulableElement::get_type() | PerBatchElement::get_type(); }

    int initialize() {
        uniform_dist = std::uniform_int_distribution<int64_t>(0, _LB_PPC_GLOBAL_MY_CPU_TIME);
        random_generator = std::default_random_engine();

        /* We have only two ranges for CPU and GPU. */
        local_cpu_ratio = 0;

        return 0;
    }
    int initialize_global() { rte_atomic64_set(&(cpu_ratio), 0); return 0; }
    int initialize_per_node() { return 0; }

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        RTE_LOG(INFO, LB, "load balancer mode: Adaptive\n");
        return 0;
    }

    int process_batch(int input_port, PacketBatch *batch)
    {
        /* Generate a random number and find the interval where it belongs to. */
        int64_t x = uniform_dist(random_generator);
        int _temp = (x > local_cpu_ratio) - 1;
        anno_set(&batch->banno, NBA_BANNO_LB_DECISION, _temp);
        return 0;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
    	next_delay = 100000;
    	int64_t temp_cpu_ratio = rte_atomic64_read(&cpu_ratio);
    	local_cpu_ratio = temp_cpu_ratio;

    	if(ctx->loc.core_id == 0)
    	{
    		double true_cpu_pkt_time = ctx->inspector->true_process_time_cpu;
    		double true_gpu_pkt_time = ctx->inspector->true_process_time_gpu[0];
    		double diff = std::abs(true_cpu_pkt_time - true_gpu_pkt_time) / ((true_cpu_pkt_time + true_gpu_pkt_time) / 2);

    		if(diff > 0.05)
    		{
    			if(true_cpu_pkt_time > true_gpu_pkt_time)
    				temp_cpu_ratio -= _LB_PPC_GLOBAL_MY_CPU_DELTA;
    			else //if(diff > 0.1)
    				temp_cpu_ratio += _LB_PPC_GLOBAL_MY_CPU_DELTA;

    			if(temp_cpu_ratio > _LB_PPC_GLOBAL_MY_CPU_TIME-_LB_PPC_GLOBAL_MY_CPU_DELTA)
    				temp_cpu_ratio = _LB_PPC_GLOBAL_MY_CPU_TIME-_LB_PPC_GLOBAL_MY_CPU_DELTA;
    			if(temp_cpu_ratio < _LB_PPC_GLOBAL_MY_CPU_DELTA)
    				temp_cpu_ratio = _LB_PPC_GLOBAL_MY_CPU_DELTA;


    			printf("CPU_:%f GPU_:%f = Ratio: %ld\n", true_cpu_pkt_time, true_gpu_pkt_time, temp_cpu_ratio);
    			rte_atomic64_set(&cpu_ratio, temp_cpu_ratio);
    		}
    	}

    	out_batch = nullptr;
        return 0;
    }

private:
    std::uniform_int_distribution<int64_t> uniform_dist;
    std::default_random_engine random_generator;

    static rte_atomic64_t cpu_ratio;
    int64_t local_cpu_ratio;
};

rte_atomic64_t LoadBalanceAdaptiveGlobal::cpu_ratio;

EXPORT_ELEMENT(LoadBalanceAdaptiveGlobal);

}

#endif

// vim: ts=8 sts=4 sw=4 et
