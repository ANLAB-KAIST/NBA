#include "None.hh"
#include "../../lib/types.hh"

using namespace std;
using namespace nba;

int None::initialize()
{
    return 0;
}

int None::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    return 0;
}

int None::process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
