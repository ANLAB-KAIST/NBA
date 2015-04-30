#include "DecIPTTL.hh"
#include "../../lib/types.hh"

using namespace std;
using namespace nba;

int DecIPTTL::initialize()
{
    return 0;
}

int DecIPTTL::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    return 0;
}

int DecIPTTL::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    struct iphdr *iph   = (struct iphdr *)(ethh + 1);
    uint32_t sum;

    if (iph->ttl <= 1)
        return DROP;

    // Decrement TTL.
    iph->ttl --;
    sum = (~ntohs(iph->check) & 0xFFFF) + 0xFEFF;
    iph->check = ~htons(sum + (sum >> 16));
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
