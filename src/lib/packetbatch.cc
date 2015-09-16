#include <nba/element/packetbatch.hh>
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_branch_prediction.h>

using namespace std;
using namespace nba;

/* empty currently */
namespace nba {

void PacketBatch::collect_excluded_packets()
{
    unsigned dropped_cnt = 0;
    for (unsigned p = 0; p < this->count - dropped_cnt; p++) {
        if (unlikely(this->excluded[p])) {
            unsigned q = this->count - dropped_cnt - 1;
            struct rte_mbuf *t = this->packets[p];
            this->packets[p] = this->packets[q];
            this->packets[q] = t;
            this->excluded[p] = false;
            this->excluded[q] = true;
            this->results[p] = this->results[q];
            this->results[q] = -1;
            dropped_cnt ++;
        }
    }
    this->count -= dropped_cnt;
    this->drop_count += dropped_cnt;  /* will be zeroed by ElementGraph. */
}

void PacketBatch::clean_drops(struct rte_ring *drop_queue)
{
    if (this->drop_count > 0) {
        rte_ring_enqueue_bulk(drop_queue,
                              (void **) &this->packets[this->count],
                              this->drop_count);
        this->drop_count = 0;
    }
}

}

// vim: ts=8 sts=4 sw=4 et
