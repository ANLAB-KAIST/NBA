#include "L2ForwardCreate.hh"
#include <nba/framework/threadcontext.hh>
#include <cassert>
#include <rte_ether.h>

using namespace std;
using namespace nba;

int L2ForwardCreate::initialize()
{
    return 0;
}

int L2ForwardCreate::configure(comp_thread_context *ctx, std::vector<std::string> &args)
{
    Element::configure(ctx, args);
    mode = 0;
    return 0;
}

int L2ForwardCreate::process(int input_port, Packet *pkt)
{
    struct ether_hdr *ethh = (struct ether_hdr *) pkt->data();
    if (likely(is_unicast_ether_addr(&ethh->d_addr))) {

        unsigned iface_in = anno_get(&pkt->anno, NBA_ANNO_IFACE_IN);
        struct ether_hdr* header = (struct ether_hdr*) pkt->data();
        struct ether_addr temp;
        header->s_addr = ctx->io_ctx->tx_ports[iface_in].addr;
        header->d_addr = header->s_addr;

        ctx->io_tx_new(header, pkt->length(), iface_in);

    }
    return DROP;
}

// vim: ts=8 sts=4 sw=4 et
