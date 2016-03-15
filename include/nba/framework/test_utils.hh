#ifndef __NBA_TEST_UTILS_HH__
#define __NBA_TEST_UTILS_HH__

#include <cstdint>
#include <functional>

struct rte_mbuf;

namespace nba {

struct Packet;
class PacketBatch;

namespace testing {

typedef std::function<void(size_t pkt_idx, struct Packet *p)>
    pkt_init_callback_t;

extern PacketBatch *create_batch(size_t num_pkts, size_t pkt_size,
                                 pkt_init_callback_t init_cb);

extern void free_batch(PacketBatch *batch);

} // endns(testing)
} // endns(nba)

#endif

// vim: ts=8 sts=4 sw=4 et
