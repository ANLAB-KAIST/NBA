#ifndef __NBA_ELEMENT_LOADBALANCEPPC_HH__
#define __NBA_ELEMENT_LOADBALANCEPPC_HH__

#include <nba/element/element.hh>
#include <nba/element/annotation.hh>
#include <nba/framework/loadbalancer.hh>
#include <nba/framework/logging.hh>
#include <nba/core/queue.hh>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <rte_errno.h>
#include <rte_atomic.h>

#define LB_PPC_CPU_RATIO_MULTIPLIER (1000)
#define LB_PPC_CPU_RATIO_DELTA (50)

namespace nba {

class LoadBalancePPC : public SchedulableElement, PerBatchElement {
public:
    LoadBalancePPC() : SchedulableElement(), PerBatchElement()
    { }

    virtual ~LoadBalancePPC()
    { }

    const char *class_name() const { return "LoadBalancePPC"; }
    const char *port_count() const { return "1/1"; }
    int get_type() const { return SchedulableElement::get_type() | PerBatchElement::get_type(); }

    int initialize()
    {
        local_cpu_ratio = LB_PPC_CPU_RATIO_MULTIPLIER;
        delta = LB_PPC_CPU_RATIO_DELTA;
        last_estimated_ppc = 0;
        rep = 0;
        rep_limit = ctx->num_coproc_ppdepth;
        initial_converge = true;
        offload = false;
        cpu_ratio = (rte_atomic64_t *) ctx->node_local_storage->get_alloc("LBPPC.cpu_weight");
        return 0;
    }

    int initialize_global() { return 0; }

    int initialize_per_node()
    {
        ctx->node_local_storage->alloc("LBPPC.cpu_weight", sizeof(rte_atomic64_t));
        rte_atomic64_t *node_cpu_ratio = (rte_atomic64_t *)
                ctx->node_local_storage->get_alloc("LBPPC.cpu_weight");
        assert(node_cpu_ratio != nullptr);
        rte_atomic64_set(node_cpu_ratio, 0);
        return 0;
    }

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        RTE_LOG(INFO, LB, "load balancer mode: Adaptive PPC\n");
        return 0;
    }

    int process_batch(int input_port, PacketBatch *batch)
    {
        int decision = 0;
        const float c = (float) local_cpu_ratio / LB_PPC_CPU_RATIO_MULTIPLIER;
        rep ++;
        if (offload) {
            decision = 1;
            if (rep == rep_limit) { // Change to CPU-mode
                if (local_cpu_ratio == 0)
                    rep_limit = 0; // only once for sampling!
                else
                    rep_limit = (unsigned) (c * ctx->num_coproc_ppdepth / (1.0f - c));
                rep = 0;
                offload = false;
            }
        } else {
            decision = 0;
            if (rep == rep_limit) { // Change to GPU-mode
                rep_limit = ctx->num_coproc_ppdepth;
                rep = 0;
                offload = true;
            }
        }
        anno_set(&batch->banno, NBA_BANNO_LB_DECISION, decision);
        return 0;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
        next_delay = 200000; // 0.2sec

        if (ctx->loc.local_thread_idx == 0) {
            const float ppc_cpu = ctx->inspector->pkt_proc_cycles[0];
            const float ppc_gpu = ctx->inspector->pkt_proc_cycles[1];
            if (initial_converge) {
                if (ppc_cpu != 0 && ppc_gpu != 0) {
                    int64_t temp_cpu_ratio;
                    if (ppc_cpu > ppc_gpu) temp_cpu_ratio = 0;
                    else temp_cpu_ratio = LB_PPC_CPU_RATIO_MULTIPLIER;
                    printf("[PPC:%d] Initial converge: %ld | CPU %'8.0f GPU %'8.0f\n", ctx->loc.node_id,
                           temp_cpu_ratio, ppc_cpu, ppc_gpu);
                    rte_atomic64_set(cpu_ratio, temp_cpu_ratio);
                    initial_converge = false;
                }
            } else {
                int64_t temp_cpu_ratio = rte_atomic64_read(cpu_ratio);
                const float estimated_ppc = (temp_cpu_ratio * ppc_cpu
                                               + (LB_PPC_CPU_RATIO_MULTIPLIER - temp_cpu_ratio) * ppc_gpu)
                                              / LB_PPC_CPU_RATIO_MULTIPLIER;
                const float c = (float) temp_cpu_ratio / LB_PPC_CPU_RATIO_MULTIPLIER;

                if (last_estimated_ppc != 0) {
                    if (last_estimated_ppc > estimated_ppc) {
                        // keep direction
                    } else {
                        // reverse direction
                        delta = -delta;
                    }
                    temp_cpu_ratio += delta;
                }
                if (temp_cpu_ratio < 0) temp_cpu_ratio = 0;
                if (temp_cpu_ratio > LB_PPC_CPU_RATIO_MULTIPLIER) temp_cpu_ratio = LB_PPC_CPU_RATIO_MULTIPLIER;
                last_estimated_ppc = estimated_ppc;

                printf("[PPC:%d] CPU %'8.0f GPU %'8.0f PPC %'8.0f CPU-Ratio %.3f\n",
                       ctx->loc.node_id,
                       ppc_cpu, ppc_gpu, estimated_ppc, c);
                rte_atomic64_set(cpu_ratio, temp_cpu_ratio);
            }
        }

        out_batch = nullptr;
        return 0;
    }

private:
    rte_atomic64_t *cpu_ratio;
    float last_estimated_ppc;
    int64_t local_cpu_ratio;
    int64_t delta;
    unsigned rep, rep_limit;
    bool initial_converge;
    bool offload;
};

EXPORT_ELEMENT(LoadBalancePPC);

}

#endif

// vim: ts=8 sts=4 sw=4 et
