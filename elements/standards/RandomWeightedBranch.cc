#include "RandomWeightedBranch.hh"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <nba/core/enumerate.hh>

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
    float sum = 0.0f;
    if (args.size() > max_num_args)
        throw invalid_argument("You cannot set too many arguments.");
    for (auto& arg : args) {
        float p = stof(arg);
        sum += p;
        out_probs.push_back(sum);
    }
    /* Example input: 0.2, 0.8
     * Example out_probs:
     * (0, 0.2)
     * (1, 1.0)
     */
    if (!approx_equal(sum, 1.0f))
        throw invalid_argument("The sum of output probabilities must be exactly 1.0 (100%).");
    return 0;
}

int RandomWeightedBranch::process(int input_port, Packet *pkt)
{
    float x = uniform_dist(random_generator);
    int last_idx = 0;
    for (auto&& pair : enumerate(out_probs)) {
        if (x < pair.second) {
            output(pair.first).push(pkt);
            return 0;
        }
        last_idx = pair.first;
    }
    /* Safeguard. */
    output(last_idx).push(pkt);
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
