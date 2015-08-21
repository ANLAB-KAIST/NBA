#ifndef __NBA_ELEMENT_LOADBALANCEBYWEIGHT_HH__
#define __NBA_ELEMENT_LOADBALANCEBYWEIGHT_HH__

#include <nba/element/element.hh>
#include <nba/element/annotation.hh>
#include <nba/framework/logging.hh>
#include <vector>
#include <string>
#include <exception>
#include <random>
#include <rte_errno.h>

namespace nba {

class LoadBalanceByWeight : public SchedulableElement, PerBatchElement {
public:
    LoadBalanceByWeight() : SchedulableElement(), PerBatchElement()
    { }

    virtual ~LoadBalanceByWeight()
    { }

    const char *class_name() const { return "LoadBalanceByWeight"; }
    const char *port_count() const { return "1/1"; }
    int get_type() const { return SchedulableElement::get_type() | PerBatchElement::get_type(); }

    int initialize() { 
        /* Initialize random engines. */
        out_probs = std::vector<float>();
        uniform_dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
        random_generator = std::default_random_engine();

        /* We have only two ranges for CPU and GPU. */
        out_probs.push_back(cpu_weight);        // cpu index in out_probs: 0
        out_probs.push_back(1.0f);                  // gpu index in out_probs: 1
        return 0;
    }

    int initialize_global() { return 0; }
    int initialize_per_node() { return 0; }

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        if (args.size() != 1)
            rte_panic("LoadBalancerByWeight: too many or few arguments. (expected: 1)\n");

        std::string num_str;
        if (args[0] == "from-env") {
            const char *env = getenv("NBA_LOADBALANCER_CPU_RATIO");
            if (env == nullptr) {
                RTE_LOG(WARNING, LB, "LoadBalanceByWeight: env-var NBA_LOADBALANCER_CPU_RATIO is not set. Falling back to CPU-only...\n");
                num_str = "1.0";
            } else {
                num_str = env;
            }
        } else {
            num_str = args[0];
        }

        try {
            cpu_weight = std::stof(num_str, nullptr);
            if (cpu_weight < 0.0f || cpu_weight > 1.0f)
                throw std::out_of_range("cpu_weight");
            gpu_weight = 1.0f - cpu_weight;
        } catch (std::out_of_range &e) {
            rte_panic("LoadBalanceByWeight: out of range (%s).\n", num_str.c_str());
        } catch (std::invalid_argument &e) {
            rte_panic("LoadBalanceByWeight: invalid argument (%s).\n", num_str.c_str());
        }

        RTE_LOG(INFO, LB, "load balancer mode: Weighted (CPU: %.2f, GPU: %.2f)\n", cpu_weight, gpu_weight);
        return 0;
    }

    int process_batch(int input_port, PacketBatch *batch)
    {
        float x = uniform_dist(random_generator);
        int idx = 0;
        int64_t choice;
        for (float cur : out_probs) {
            if (x < cur)
                break;
            idx ++;
        }
        choice = idx;
        assert(choice >= 0);
        anno_set(&batch->banno, NBA_BANNO_LB_DECISION, choice);
        return 0;
    }

    int dispatch(uint64_t loop_count, PacketBatch*& out_batch, uint64_t &next_delay)
    {
        next_delay = 1e6L;
        out_batch = nullptr;
        return 0;
    }

private:
    std::vector<float> out_probs;
    std::uniform_real_distribution<float> uniform_dist;
    std::default_random_engine random_generator;

    float cpu_weight;
    float gpu_weight;
};

EXPORT_ELEMENT(LoadBalanceByWeight);

}

#endif

// vim: ts=8 sts=4 sw=4 et
