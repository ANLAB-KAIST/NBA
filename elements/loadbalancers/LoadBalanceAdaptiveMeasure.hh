#ifndef __NBA_ELEMENT_LOADBALANCEADAPTIVEMEASURE_HH__
#define __NBA_ELEMENT_LOADBALANCEADAPTIVEMEASURE_HH__

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include "../../lib/loadbalancer.hh"
#include "../../lib/queue.hh"

#include <rte_errno.h>
#include <rte_log.h>
#include <rte_approx.h>
#include <rte_atomic.h>

#include <vector>
#include <string>
#include <random>
#include <unistd.h>

#define LB_MEASURE_CPU_RATIO_MULTIPLIER (1000)
#define LB_MEASURE_CPU_RATIO_DELTA (50)
#define LB_MEASURE_REPTITON_PER_RATIO (16)

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

    int initialize()
    {
        /* We have only two ranges for CPU and GPU. */
        local_cpu_ratio = 0;
        print_count = 1;
        rep = 0;
        rep_limit = ctx->num_coproc_ppdepth;
        offload = false;
        cpu_ratio = (rte_atomic64_t *) ctx->node_local_storage->get_alloc("LBMeasure.cpu_weight");

        uniform_dist = std::uniform_int_distribution<int64_t>(0, LB_MEASURE_CPU_RATIO_MULTIPLIER);
        random_generator = std::default_random_engine();
        return 0;
    }

    int initialize_global() { return 0; }

    int initialize_per_node()
    {
        ctx->node_local_storage->alloc("LBMeasure.cpu_weight", sizeof(rte_atomic64_t));
        rte_atomic64_t *node_cpu_ratio = (rte_atomic64_t *)
                ctx->node_local_storage->get_alloc("LBMeasure.cpu_weight");
        assert(node_cpu_ratio != nullptr);
        rte_atomic64_set(node_cpu_ratio, 0);
        return 0;
    }

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        RTE_LOG(INFO, LB, "load balancer mode: Measure\n");
        return 0;
    }

    int process_batch(int input_port, PacketBatch *batch)
    {
#if 0
        // {{{ Randomized load balancer
        int64_t x = uniform_dist(random_generator);
        int _temp = (x > local_cpu_ratio);
        anno_set(&batch->banno, NBA_BANNO_LB_DECISION, _temp);
        return 0;
        // }}}
#endif
        // {{{ Deterministic load balancer
        int decision = 0;
        rep ++;
        if (offload) { // GPU-mode
            decision = 1;
            if (rep >= rep_limit_gpu) { // Change to CPU-mode
                if (local_cpu_ratio == 0)
                    rep_limit = 0; // only once for sampling!
                else
                    rep_limit = rep_limit_cpu;
                rep = 0;
                offload = false;
            }
        } else {       // CPU-mode
            decision = 0;
            if (rep >= rep_limit_cpu) { // Change to GPU-mode
                rep_limit = rep_limit_gpu;
                rep = 0;
                offload = true;
            }
        }
        //printf("rep %u, offload %d, rlcpu %u, rlgpu %u\n", rep, decision, rep_limit_cpu, rep_limit_gpu);
        // }}}
        anno_set(&batch->banno, NBA_BANNO_LB_DECISION, decision);
        return 0;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
        next_delay = 200000; // 0.2sec
        int64_t temp_cpu_ratio = rte_atomic64_read(cpu_ratio);
        local_cpu_ratio = temp_cpu_ratio;
        const float c = (float) temp_cpu_ratio / LB_MEASURE_CPU_RATIO_MULTIPLIER;

        if (ctx->io_ctx->loc.local_thread_idx == 0) {
            const float ppc_cpu = ctx->inspector->pkt_proc_cycles[0];
            const float ppc_gpu = ctx->inspector->pkt_proc_cycles[1];
            const float estimated_ppc = (temp_cpu_ratio * ppc_cpu
                                           + (LB_MEASURE_CPU_RATIO_MULTIPLIER - temp_cpu_ratio) * ppc_gpu)
                                          / LB_MEASURE_CPU_RATIO_MULTIPLIER;
            printf("[MEASURE:%d] CPU %'8.0f GPU %'8.0f PPC %'8.0f CPU-Ratio %.3f (cpu_rep_limit %u)\n",
                   ctx->loc.node_id,
                   ppc_cpu, ppc_gpu, estimated_ppc, c,
                   (unsigned) (c * ctx->num_coproc_ppdepth / (1.0f - c)));

            if ((print_count++) % LB_MEASURE_REPTITON_PER_RATIO == 0) {
                temp_cpu_ratio += LB_MEASURE_CPU_RATIO_DELTA;
                if (temp_cpu_ratio > LB_MEASURE_CPU_RATIO_MULTIPLIER - LB_MEASURE_CPU_RATIO_DELTA) {
                    temp_cpu_ratio = LB_MEASURE_CPU_RATIO_MULTIPLIER - LB_MEASURE_CPU_RATIO_DELTA;
                    printf("END_OF_TEST\n");
                    raise(SIGINT);
                }
                rte_atomic64_set(cpu_ratio, temp_cpu_ratio);
            }
        }
        #if 1
        rep_limit_cpu = (unsigned) (c * ctx->num_coproc_ppdepth / (1.0f - c));
        rep_limit_gpu = ctx->num_coproc_ppdepth;
        #else
        {
            uint32_t p, q;
            if (local_cpu_ratio == 0) {
                rep_limit_cpu = 1;
                rep_limit_gpu = ctx->num_coproc_ppdepth;
            } else if (local_cpu_ratio == LB_MEASURE_CPU_RATIO_MULTIPLIER) {
                rep_limit_cpu = ctx->num_coproc_ppdepth;
                rep_limit_gpu = 1;
            } else {
                assert(0 == rte_approx((double) local_cpu_ratio / LB_MEASURE_CPU_RATIO_MULTIPLIER,
                                       0.02, &p, &q));
                rep_limit_cpu = p;
                rep_limit_gpu = q - p;
            }
        }
        #endif
        printf("[MEASURE-REPLIM:%d] %ld; %u %u\n", ctx->loc.node_id, local_cpu_ratio, rep_limit_cpu, rep_limit_gpu);

        out_batch = nullptr;
        return 0;
    }

private:
    rte_atomic64_t *cpu_ratio;
    int64_t local_cpu_ratio;
    int print_count;

    unsigned rep, rep_limit;
    unsigned rep_limit_cpu, rep_limit_gpu;
    bool offload;

    std::uniform_int_distribution<int64_t> uniform_dist;
    std::default_random_engine random_generator;
};

EXPORT_ELEMENT(LoadBalanceAdaptiveMeasure);

}

#endif

// vim: ts=8 sts=4 sw=4 et foldmethod=marker
