#ifndef __LB_WEIGHTEDRANDOM_2_HH__
#define __LB_WEIGHTEDRANDOM_2_HH__

#include "../lib/loadbalancer.hh"
#include "../lib/config.hh"
#include <cmath>
#include <random>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace std;

namespace nshader {

static inline bool approx_equal(float a, float b)
{
    return fabs(a - b) < numeric_limits<float>::epsilon();
}

// WeightedRandom version 2: Refered RandomWeightedBranch element.
class WeightedRandomLB2 : public LoadBalancer
{
public:
    WeightedRandomLB2(): LoadBalancer() {
        cpu_weight = (float) load_balancer_cpu_ratio;
        gpu_weight = 1.0f - cpu_weight;

        // Init!
        out_probs = std::vector<float>();
        uniform_dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
        random_generator = std::default_random_engine();

        // Quick-and-dirty
        out_probs.push_back(cpu_weight);        // cpu index in out_probs: 0
        out_probs.push_back(1.0f);                  // gpu index in out_probs: 1
        /*
        // Generalized ver.
        float total_sum = 0.0f;
        for (auto it = args.begin(); it != args.end(); it++) {
            float p = stof(*it) / 100.0f;
            total_sum += p;
        }

        float sum = 0.0f;
        for (auto it = args.begin(); it != args.end(); it++) {
            float p = stof(*it) / 100.0f;
            sum += p;
            out_probs.push_back(sum/total_sum);
        }

        if (!approx_equal(sum/total_sum, 1.0f))
            throw invalid_argument("The sum of output probabilities must be exactly 1 (100%).");
        */
    };
    virtual ~WeightedRandomLB2() { };

    int gate_keeper(PacketBatch *batch, vector<ComputeDevice*>& devices)
    {
        float x = uniform_dist(random_generator);
        int idx = 0;
        int choice;
        for (auto cur = out_probs.begin(); cur != out_probs.end(); cur++) {
            if(x < *cur)
                return idx-1;
            idx++;
        }
        choice = idx-1;
        assert(choice != -2);
        return choice;
    };

    uint64_t update_params(SystemInspector &inspector, uint64_t timestamp)
    {
        // do nothing...
        return 1000000u;
    };

    std::vector<float> out_probs;
    std::uniform_real_distribution<float> uniform_dist;
    std::default_random_engine random_generator;

    unsigned cpu_count;
    unsigned gpu_count;
    float cpu_weight;
    float gpu_weight;
};

}

EXPORT_LOADBALANCER(WeightedRandomLB2);

#endif

// vim: ts=8 sts=4 sw=4 et
