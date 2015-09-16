#include "RandomWeightedBranch.hh"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace std;
using namespace nba;

static inline bool approx_equal(float a, float b)
{
    return fabs(a - b) < numeric_limits<float>::epsilon();
}

int RandomWeightedBranch::initialize()
{
    return 0;
}

int RandomWeightedBranch::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
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
    return 0;
}

int RandomWeightedBranch::process(int input_port, Packet *pkt)
{
    float x = uniform_dist(random_generator);
    int idx = 0;
    for (auto cur = out_probs.begin(); cur != out_probs.end(); cur++) {
        if(x < *cur) {
            output(idx).push(pkt);
            return 0;
        }
        idx++;
    }
    output(idx - 1).push(pkt);
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
