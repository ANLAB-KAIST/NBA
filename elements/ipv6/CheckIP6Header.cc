#include "CheckIP6Header.hh"
#include "../../lib/types.hh"

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

int CheckIP6Header::process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno)
{
    struct ether_hdr *ethh = rte_pktmbuf_mtod(pkt, struct ether_hdr *);
    struct ip6_hdr *iph = (struct ip6_hdr *)(ethh + 1);

    // Validate the packet header.
    if (ntohs(ethh->ether_type) != ETHER_TYPE_IPv6) {
        //RTE_LOG(DEBUG, ELEM, "CheckIP6Header: invalid packet type - %x\n", ntohs(ethh->ether_type));
        return DROP;
    }

    //if ((iph->ip6_vfc & 0xf0) >> 4 != 6)  // get the first 4 bits.
    //    return DROP;

    // TODO: Discard illegal source addresses.

    return 0; // output port number: 0
}

// vim: ts=8 sts=4 sw=4 et
