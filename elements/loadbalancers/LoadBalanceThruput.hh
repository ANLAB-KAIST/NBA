#ifndef __NBA_ELEMENT_LOADBALANCETHRUPUT_HH__
#define __NBA_ELEMENT_LOADBALANCETHRUPUT_HH__

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
#include <cmath>

namespace nba {

class LoadBalanceThruput : public SchedulableElement, PerBatchElement {
public:
    LoadBalanceThruput() : SchedulableElement(), PerBatchElement(), direction(0), thruput_history()
    { }

    virtual ~LoadBalanceThruput()
    { }

    const char *class_name() const { return "LoadBalanceThruput"; }
    const char *port_count() const { return "1/1"; }
    int get_type() const { return SchedulableElement::get_type() | PerBatchElement::get_type(); }

    int initialize() {
        uniform_dist = std::uniform_int_distribution<int64_t>(0, 1000);
        random_generator = std::default_random_engine();

        local_cpu_ratio = 1000;
        direction = 1;
        num_pass = 2;
        rte_atomic64_set(&cpu_ratio, 1000);
        rte_atomic64_set(&update_count, 0);

        thruput_history.init(3, ctx->loc.node_id);
        last_count = 0;
        last_thruput = 0;
        last_cpu_ratio_update = 0.1;
        last_direction_changed = get_usec();

        return 0;
    }
    int initialize_global() { return 0; }
    int initialize_per_node() { return 0; }

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        RTE_LOG(INFO, LB, "load balancer mode: Thruput\n");
        return 0;
    }

    int process_batch(int input_port, PacketBatch *batch)
    {
        /* Generate a random number and find the interval where it belongs to. */
        int64_t x = uniform_dist(random_generator);
        int choice = (x > local_cpu_ratio);
        anno_set(&batch->banno, NBA_BANNO_LB_DECISION, choice);
        return 0;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
        int64_t temp_cpu_ratio = rte_atomic64_read(&cpu_ratio);
        local_cpu_ratio = temp_cpu_ratio;
        /* Ensure that other threads have applied the new ratio. */

        rte_atomic64_inc(&update_count);

        //printf("LB: uc %ld uf %ld\n", update_count.cnt, update_flag.cnt);

        if (ctx->io_ctx->loc.node_id == 1 && ctx->io_ctx->loc.local_thread_idx == 0) {
            if (rte_atomic64_read(&update_count) > ctx->io_ctx->num_io_threads) {
                rte_atomic64_clear(&update_count);
                rte_atomic64_inc(&update_flag);
            }
            if (!rte_atomic64_cmpset((volatile uint64_t *) &update_flag.cnt, num_pass, 0))
                goto skip;

            //if (/* TODO: there is no drop */) {
            //    temp_cpu_ratio = 1000;
            //}

            #define LB_THRUPUT_DELTA 2
            //double now_thruput = ctx->inspector->tx_pkt_thruput;
            //double now_thruput = (last_thruput * 3 + (double) ctx->io_ctx->tx_pkt_thruput) / 4;
            double now_thruput = (double) ctx->io_ctx->tx_pkt_thruput;
            bool ignore_update = false;
            if (now_thruput > last_thruput) {
                /* If the thruput is surely increasing, keep decision. */
                //direction = direction;
                if (direction < 0) {
                    direction -= 1;
                    if (direction == 0) direction = -1;
                    if (direction < -LB_THRUPUT_DELTA) direction = -LB_THRUPUT_DELTA;
                } else {
                    direction += 1;
                    if (direction == 0) direction = 1;
                    if (direction > LB_THRUPUT_DELTA) direction = LB_THRUPUT_DELTA;
                }
            } else if (now_thruput <= last_thruput) {
                //direction = -1 * direction;
                /* Change direction */
                if (direction < 0) {
                    direction += 1;
                    if (direction == 0) direction = 1;
                } else {
                    direction -= 1;
                    if (direction == 0) direction = -1;
                }
            } else
                ignore_update = true;
            /* If direction is positive, increase CPU weight.
             * Otherwise, increase GPU weight. */
            assert(direction != 0);
            double cpu_ratio_update = direction * 20;
            temp_cpu_ratio += cpu_ratio_update;
            last_cpu_ratio_update = cpu_ratio_update;
            if (temp_cpu_ratio < 200) {
                num_pass = 32;
                //ctx->io_ctx->LB_THRUPUT_WINDOW_SIZE = 1638400;
            } else if (temp_cpu_ratio < 500) {
                num_pass = 16;
                //ctx->io_ctx->LB_THRUPUT_WINDOW_SIZE = 163840;
            } else if (temp_cpu_ratio < 700) {
                num_pass = 8;
                //ctx->io_ctx->LB_THRUPUT_WINDOW_SIZE = 16384;
            } else {
                num_pass = 2;
                //ctx->io_ctx->LB_THRUPUT_WINDOW_SIZE = 4096;
            }

            if (temp_cpu_ratio < 0) { temp_cpu_ratio = 0; direction = 1; }
            if (temp_cpu_ratio > 1000) { temp_cpu_ratio = 1000; direction = -1; }
            //if (temp_cpu_ratio < 0) { temp_cpu_ratio = 0; direction = 1 * LB_THRUPUT_DELTA; }
            //if (temp_cpu_ratio > 1000) { temp_cpu_ratio = 1000; direction = -1 * LB_THRUPUT_DELTA; }
            rte_atomic64_set(&cpu_ratio, temp_cpu_ratio);

            printf("ALB[%u]@%lu: temp_cpu_ratio %4ld now_thruput %f, cpu_ratio_update %f, direction %d\n", ctx->loc.core_id, get_usec(),
                   temp_cpu_ratio,
                   now_thruput,
                   cpu_ratio_update,
                   direction);

            last_thruput = now_thruput;
        }
        skip:

        next_delay = 2e5L;
        out_batch = nullptr;  /* This is a pure schedulable method, not an entry point. */

        return 0;
    }

private:
    int direction;

    uint64_t last_count;
    double last_cpu_ratio_update;
    double last_thruput;

    uint64_t last_direction_changed;
    FixedRing<uint64_t, 0> thruput_history;

    static rte_atomic64_t cpu_ratio __rte_cache_aligned;
    static rte_atomic64_t update_flag;
    static rte_atomic64_t update_count __rte_cache_aligned;

    int64_t local_cpu_ratio;
    unsigned num_pass;

    std::vector<float> out_probs;
    std::uniform_int_distribution<int64_t> uniform_dist;
    std::default_random_engine random_generator;

    float cpu_weight;
};

EXPORT_ELEMENT(LoadBalanceThruput);

rte_atomic64_t LoadBalanceThruput::cpu_ratio;
rte_atomic64_t LoadBalanceThruput::update_flag;
rte_atomic64_t LoadBalanceThruput::update_count;

}

#endif

// vim: ts=8 sts=4 sw=4 et
