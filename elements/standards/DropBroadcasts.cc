#include "DropBroadcasts.hh"
#include "../../lib/types.hh"
#include <netinet/ip.h>

using namespace std;
using namespace nshader;

int DropBroadcasts::initialize()
{
    return 0;
}

int DropBroadcasts::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    return 0;
}

int DropBroadcasts::process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    struct ether_hdr *ethh = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
    if (likely(is_unicast_ether_addr(&ethh->d_addr))) {
        return 0;
    }
    return DROP;
}

// vim: ts=8 sts=4 sw=4 et
