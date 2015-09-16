#include <nba/element/packet.hh>
#include <nba/element/packetbatch.hh>
#include <cstring>
#include <rte_mbuf.h>
#include <rte_mempool.h>

using namespace nba;

namespace nba {
thread_local struct rte_mempool *packet_pool = nullptr;

static void packet_init_packet(struct rte_mempool *mp, void *arg, void *obj, unsigned idx)
{
    return;
}

struct rte_mempool *packet_create_mempool(size_t size, int node_id, int core_id)
{
    char temp[RTE_MEMPOOL_NAMESIZE];
    snprintf(temp, RTE_MEMPOOL_NAMESIZE, "packet@%u:%u", node_id, core_id);
    assert(packet_pool == nullptr);
    packet_pool = rte_mempool_create(temp, size, sizeof(Packet), 64, 0,
                                     nullptr, nullptr,
                                     packet_init_packet, nullptr,
                                     node_id, 0);
    assert(packet_pool != nullptr);
    return packet_pool;
}

void Packet::kill()
{
    mother->results[bidx] = PacketDisposition::DROP;
    mother->excluded[bidx] = true;
    mother->has_dropped = true;
}

}

// vim: ts=8 sts=4 sw=4 et
