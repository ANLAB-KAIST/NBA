#ifndef __NBA_KNAPP_POLLRING_HH__
#define __NBA_KNAPP_POLLRING_HH__

#include <cstdint>
#include <scif.h>

#ifndef __MIC__
struct rte_ring;
#endif

namespace nba { namespace knapp {

typedef uintptr_t poll_item_t;

static_assert(sizeof(poll_item_t) <= sizeof(uint64_t), "poll items must fit within 64 bits.");

class PollRing
{
public:
#ifdef __MIC__
    PollRing(scif_epd_t epd, size_t len);
#else
    PollRing(scif_epd_t epd, size_t len, int node_id);
#endif
    virtual ~PollRing();

#ifndef __MIC__
    void get(poll_item_t &value);
    void put(const poll_item_t value);
#endif
    void wait(const unsigned idx, const poll_item_t value);
    bool poll(const unsigned idx, const poll_item_t value);
    void notify(const unsigned idx, const poll_item_t value);
    void remote_notify(const unsigned idx, const poll_item_t value);

    size_t len() const { return _len; }
    poll_item_t *get_va() const { return ring; }
    poll_item_t *get_ra() const { return ring_ra; }

private:
    scif_epd_t epd;
    size_t _len;
#ifndef __MIC__
    int node_id;
#endif

    size_t alloc_bytes;
#ifndef __MIC__
    struct rte_ring *id_pool;
#endif
    poll_item_t *ring;
    poll_item_t *ring_ra;
};

}} //endns(nba::knapp)

#endif //__NBA_KNAPP_POLLRING_HH__

// vim: ts=8 sts=4 sw=4 et
