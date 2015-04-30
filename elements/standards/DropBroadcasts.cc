#include "DropBroadcasts.hh"
#include "../../lib/types.hh"
#include <netinet/ip.h>

using namespace std;
using namespace nba;

int DropBroadcasts::initialize()
{
    return 0;
}

int DropBroadcasts::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    return 0;
}

int DropBroadcasts::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    if (likely(is_unicast_ether_addr(&ethh->d_addr))) {
        return 0;
    }
    return DROP;
}

// vim: ts=8 sts=4 sw=4 et
