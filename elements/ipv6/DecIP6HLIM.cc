#include "DecIP6HLIM.hh"
#include "../../lib/types.hh"

using namespace std;
using namespace nba;

int DecIP6HLIM::initialize()
{
    return 0;
}

int DecIP6HLIM::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    return 0;
}

int DecIP6HLIM::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    struct ip6_hdr *iph    = (struct ip6_hdr *)(ethh + 1);
    uint32_t checksum;

    if (iph->ip6_hlim <= 1) {
        return DROP;
    }

    // Decrement TTL.
    iph->ip6_hlim = htons(ntohs(iph->ip6_hlim) - 1);

    return 0;
}

// vim: ts=8 sts=4 sw=4 et
