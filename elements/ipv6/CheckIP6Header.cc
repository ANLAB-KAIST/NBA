#include "CheckIP6Header.hh"
#include <nba/framework/logging.hh>
#include <rte_ether.h>
#include <netinet/in.h>
#include <netinet/ip6.h>

using namespace std;
using namespace nba;

int CheckIP6Header::initialize()
{
    return 0;
}

int CheckIP6Header::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    return 0;
}

int CheckIP6Header::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    struct ip6_hdr *iph = (struct ip6_hdr *)(ethh + 1);

    // Validate the packet header.
    if (ntohs(ethh->ether_type) != ETHER_TYPE_IPv6) {
        pkt->kill();
        return 0;
    }

    if ((iph->ip6_vfc & 0xf0) >> 4 != 6) {  // get the first 4 bits.
        pkt->kill();
        return 0;
    }

    // TODO: Discard illegal source addresses.
    output(0).push(pkt);
    return 0; // output port number: 0
}

// vim: ts=8 sts=4 sw=4 et
