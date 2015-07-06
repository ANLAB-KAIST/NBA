#ifndef __NBA_ELEMENT_LOADBALANCEADAPTIVEMEASURE_HH__
#define __NBA_ELEMENT_LOADBALANCEADAPTIVEMEASURE_HH__

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
#include <unistd.h>

#define LB_MEASURE_CPU_RATIO_MULTIPLIER (1000)
#define LB_MEASURE_CPU_RATIO_DELTA (50)
#define LB_MEASURE_REPTITON_PER_RATIO (8)

namespace nba {

class LoadBalanceAdaptiveMeasure : public SchedulableElement, PerBatchElement {
public:
    LoadBalanceAdaptiveMeasure() : SchedulableElement(), PerBatchElement()
    { }

    virtual ~LoadBalanceAdaptiveMeasure()
    { }

    const char *class_name() const { return "LoadBalanceAdaptiveMeasure"; }
    const char *port_count() const { return "1/1"; }
    int get_type() const { return SchedulableElement::get_type() | PerBatchElement::get_type(); }

    int initialize() {
        uniform_dist = std::uniform_int_distribution<int64_t>(0, LB_MEASURE_CPU_RATIO_MULTIPLIER);
        random_generator = std::default_random_engine();

        /* We have only two ranges for CPU and GPU. */
        local_cpu_ratio = 0;
        print_count = 1;

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
        int _temp = (x > local_cpu_ratio);
        anno_set(&batch->banno, NBA_BANNO_LB_DECISION, _temp);
        return 0;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
        next_delay = 200000;
        int64_t temp_cpu_ratio = rte_atomic64_read(&cpu_ratio);
        local_cpu_ratio = temp_cpu_ratio;

        //if (ctx->io_ctx->loc.local_thread_idx == 0) {
        if (ctx->io_ctx->loc.core_id == 0) {
            double cpu_ppc = ctx->inspector->pkt_proc_cycles[0];
            double gpu_ppc = ctx->inspector->pkt_proc_cycles[1];
            double estimated_ppc = (temp_cpu_ratio * cpu_ppc
                                    + (LB_MEASURE_CPU_RATIO_MULTIPLIER - temp_cpu_ratio) * gpu_ppc)
                                   / LB_MEASURE_CPU_RATIO_MULTIPLIER;

            printf("[MEASURE:%d] CPU %12f GPU %12f PPC %12f Ratio %.3f\n", ctx->loc.node_id,
                   cpu_ppc, gpu_ppc, estimated_ppc, ((double)temp_cpu_ratio) / LB_MEASURE_CPU_RATIO_MULTIPLIER);

            if ((print_count++) % LB_MEASURE_REPTITON_PER_RATIO == 0)
            {
                temp_cpu_ratio += LB_MEASURE_CPU_RATIO_DELTA;
                if (temp_cpu_ratio > LB_MEASURE_CPU_RATIO_MULTIPLIER - LB_MEASURE_CPU_RATIO_DELTA)
                {
                    temp_cpu_ratio = LB_MEASURE_CPU_RATIO_MULTIPLIER - LB_MEASURE_CPU_RATIO_DELTA;
                    printf("END_OF_TEST\n");
                    raise(SIGINT);
                }
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
    int print_count;
};

rte_atomic64_t LoadBalanceAdaptiveMeasure::cpu_ratio;

EXPORT_ELEMENT(LoadBalanceAdaptiveMeasure);

}

#endif

// vim: ts=8 sts=4 sw=4 et
