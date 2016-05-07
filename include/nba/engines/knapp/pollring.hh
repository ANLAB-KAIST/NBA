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
    bool wait(const unsigned idx, const poll_item_t value);
    bool poll(const unsigned idx, const poll_item_t value);
    void notify(const unsigned idx, const poll_item_t value);
    void remote_notify(const unsigned idx, const poll_item_t value);

    size_t len() const { return _len; }
    poll_item_t *va() const { return _local_va; }
    poll_item_t *ra() const { return _local_ra; }
    void set_peer_ra(off_t value) { this->_peer_ra = value; }
    off_t peer_ra() const { return _peer_ra; }

private:
    scif_epd_t _epd;
    size_t _len;
    size_t _alloc_bytes;
#ifndef __MIC__
    struct rte_ring *_id_pool;
#endif
    poll_item_t *_local_va;
    poll_item_t *_local_ra;
    off_t _peer_ra;
};

}} //endns(nba::knapp)

#endif //__NBA_KNAPP_POLLRING_HH__

// vim: ts=8 sts=4 sw=4 et
