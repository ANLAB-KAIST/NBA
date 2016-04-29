#include "RandomWeightedBranch.hh"
#include <cassert>
#include <nba/core/enumerate.hh>

using namespace std;
using namespace nba;

int RandomWeightedBranch::initialize()
{
    return 0;
}

int RandomWeightedBranch::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    vector<float> weights;
    for (auto& arg : args)
        weights.push_back(stof(arg));
    ddist = discrete_distribution<int>(weights.begin(), weights.end());
    assert(ddist.min() == 0);
    assert((unsigned) (ddist.max() + 1) == args.size());
    /* Example input: 0.3, 0.5
     * Example discrete distribution:
     * (0, 0.3/(0.3+0.5))
     * (1, 0.5/(0.3+0.5))
     */
    return 0;
}

int RandomWeightedBranch::process(int input_port, Packet *pkt)
{
    int idx = ddist(gen);
    output(idx).push(pkt);
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
