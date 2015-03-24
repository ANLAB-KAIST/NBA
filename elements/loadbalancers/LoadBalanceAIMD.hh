#ifndef __NBA_ELEMENT_LOADBALANCEAIMD_HH__
#define __NBA_ELEMENT_LOADBALANCEAIMD_HH__

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include "../../lib/loadbalancer.hh"
#include "../../lib/queue.hh"
extern "C" {
#include <rte_errno.h>
#include <rte_log.h>
}
#include <vector>
#include <string>
#include <random>

namespace nba {

class LoadBalanceAIMD : public SchedulableElement, PerBatchElement {
public:
    LoadBalanceAIMD() : SchedulableElement(), PerBatchElement()
    { }

    virtual ~LoadBalanceAIMD()
    { }

    const char *class_name() const { return "LoadBalanceAIMD"; }
    const char *port_count() const { return "1/1"; }
    int get_type() const { return SchedulableElement::get_type() | PerBatchElement::get_type(); }

    int initialize() {
        out_probs = std::vector<float>();
        uniform_dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
        random_generator = std::default_random_engine();

        /* We have only two ranges for CPU and GPU. */
        cpu_weight = 1.0f;
        out_probs.push_back(cpu_weight);        // cpu index in out_probs: 0
        out_probs.push_back(1.0f);                  // gpu index in out_probs: 1

        last_thread_cpu_time = get_thread_cpu_time();
        last_usec = get_usec();
        last_loop_count = 0;

        return 0;
    }
    int initialize_global() { return 0; }
    int initialize_per_node() { return 0; }

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        RTE_LOG(INFO, LB, "load balancer mode: AIMD\n");
        return 0;
    }

    int process_batch(int input_port, PacketBatch *batch)
    {
        /* Generate a random number and find the interval where it belongs to. */
        float x = uniform_dist(random_generator);
        int idx = 0;
        int64_t choice;
        for (float cur : out_probs) {
            if (x < cur)
                break;
            idx ++;
        }
        choice = idx - 1;
        assert(choice != -2);
        anno_set(&batch->banno, NBA_BANNO_LB_DECISION, choice);
        return 0;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
        uint64_t t = get_thread_cpu_time();
        const double base_idle = 0.103;
        const uint64_t c = get_usec();
        uint64_t time_delta = c - last_usec;
        uint64_t loop_count_delta = loop_count - last_loop_count;
        double busy = (t - last_thread_cpu_time) / 1e9 - base_idle;
        //double busy = (t - last_thread_cpu_time) / time_delta / 1e3;
        //double busy = 1.0 - (loop_count_delta / 1.2e5);
        last_thread_cpu_time = t;
        last_usec = c;
        //if (ctx->loc.core_id == 16)
        //    printf("ALB[%u]: %.2f, %.3f\n", ctx->loc.core_id, cpu_weight, busy);
        last_loop_count = loop_count;

        /* ref: PID controller */
        //error = (set_point - actual_output);
        //error_sum += (error * time_delta);
        //error_diff = (error - last_error) / time_delta;
        //last_error = error;
        //cpu_weight = kp * error + ki * error_sum + kd * error_diff;

        /* AIMD controller */
        if (busy > 0.1) {
            cpu_weight /= 2;
        } else if (busy > 0.02) {
            cpu_weight -= 0.2;
        }
        cpu_weight += 0.01;
        if (cpu_weight < 0.0f) cpu_weight = 0.0f;
        if (cpu_weight > 1.0f) cpu_weight = 1.0f;

        /* Set the weight. */
        out_probs[0] = cpu_weight;

        next_delay = 1e5L;
        out_batch = nullptr;  /* This is a pure schedulable method, not an entry point. */

        return 0;
    }

private:
    uint64_t last_thread_cpu_time;
    uint64_t last_usec;
    uint64_t last_loop_count;

    std::vector<float> out_probs;
    std::uniform_real_distribution<float> uniform_dist;
    std::default_random_engine random_generator;

    float cpu_weight;
};

EXPORT_ELEMENT(LoadBalanceAIMD);

}

#endif

// vim: ts=8 sts=4 sw=4 et
