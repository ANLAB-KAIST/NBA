#include <cassert>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <nba/framework/test_utils.hh>
#include <nba/element/annotation.hh>
#include <nba/element/packet.hh>
#include <nba/element/packetbatch.hh>

using namespace std;
using namespace nba;

PacketBatch *nba::testing::create_batch
(size_t num_pkts, size_t pkt_size, pkt_init_callback_t init_cb)
{
    PacketBatch *batch = new PacketBatch();
    batch->count = num_pkts;
    INIT_BATCH_MASK(batch);
    batch->banno.bitmask = 0;
    anno_set(&batch->banno, NBA_BANNO_LB_DECISION, -1);
    #if NBA_BATCHING_SCHEME == NBA_BATCHING_LINKEDLIST
    batch->first_idx = 0;
    batch->last_idx = batch->count - 1;
    batch->slot_count = batch->count;
    Packet *prev_pkt = nullptr;
    #endif
    for (unsigned pkt_idx = 0; pkt_idx < num_pkts; pkt_idx++) {
        batch->packets[pkt_idx] = (struct rte_mbuf *) malloc(sizeof(struct rte_mbuf)
                                                             + RTE_PKTMBUF_HEADROOM
                                                             + NBA_MAX_PACKET_SIZE);
    }
    FOR_EACH_PACKET_ALL_INIT_PREFETCH(batch, 8u) {
        assert(pkt_idx < num_pkts);
        assert(nullptr != batch->packets[pkt_idx]);
        batch->packets[pkt_idx]->nb_segs = 1;
        batch->packets[pkt_idx]->buf_addr = (void *) ((uintptr_t) batch->packets[pkt_idx]
                                                      + sizeof(struct rte_mbuf));
        batch->packets[pkt_idx]->data_off = RTE_PKTMBUF_HEADROOM;
        batch->packets[pkt_idx]->port = 0;
        batch->packets[pkt_idx]->pkt_len = pkt_size;
        batch->packets[pkt_idx]->data_len = pkt_size;
        Packet *pkt = Packet::from_base_nocheck(batch->packets[pkt_idx]);
        new (pkt) Packet(batch, batch->packets[pkt_idx]);
        #if NBA_BATCHING_SCHEME == NBA_BATCHING_LINKEDLIST
        if (prev_pkt != nullptr) {
            prev_pkt->next_idx = pkt_idx;
            pkt->prev_idx = pkt_idx - 1;
        }
        prev_pkt = pkt;
        #endif
        memset(pkt->data(), 0, pkt_size);
        pkt->anno.bitmask = 0;
        anno_set(&pkt->anno, NBA_ANNO_IFACE_IN,
                 batch->packets[pkt_idx]->port);
        anno_set(&pkt->anno, NBA_ANNO_TIMESTAMP, 1234);
        anno_set(&pkt->anno, NBA_ANNO_BATCH_ID, 10000);
        init_cb(pkt_idx, pkt);
    } END_FOR_ALL_INIT_PREFETCH;

    return batch;
}

void nba::testing::free_batch(PacketBatch *batch)
{
    if (batch->datablock_states != nullptr) {
        if (batch->datablock_states->aligned_item_sizes_h.ptr != nullptr)
            free(batch->datablock_states->aligned_item_sizes_h.ptr);
        delete batch->datablock_states;
    }
    for (unsigned pkt_idx = 0; pkt_idx < batch->count; pkt_idx++) {
        free(batch->packets[pkt_idx]);
    }
    delete batch;
}

// vim: ts=8 sts=4 sw=4 et
