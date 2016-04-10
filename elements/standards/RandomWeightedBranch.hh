#ifndef __NBA_ELEMENT_RANDOMWEIGHTEDBRANCH_HH__
#define __NBA_ELEMENT_RANDOMWEIGHTEDBRANCH_HH__

#include <nba/element/element.hh>
#include <nba/core/queue.hh>
#include <vector>
#include <string>
#include <random>

namespace nba {

class RandomWeightedBranch : public Element {
public:
    RandomWeightedBranch(): Element(), out_probs(),
                    uniform_dist(0.0f, 1.0f), random_generator()
    {
    }

    ~RandomWeightedBranch()
    {
    }

    const char *class_name() const { return "RandomWeightedBranch"; }
    const char *port_count() const { return "1/*"; }

    int initialize();
    int initialize_global() { return 0; };      // per-system configuration
    int initialize_per_node() { return 0; };    // per-node configuration
    int configure(comp_thread_context *ctx, std::vector<std::string> &args);

    int process(int input_port, Packet *pkt);

private:
    static const size_t max_num_args = 10;
    FixedArray<float, max_num_args> out_probs;
    std::uniform_real_distribution<float> uniform_dist;
    std::default_random_engine random_generator;
} __attribute__ ((aligned(64)));

EXPORT_ELEMENT(RandomWeightedBranch);

}

#endif

// vim: ts=8 sts=4 sw=4 et
