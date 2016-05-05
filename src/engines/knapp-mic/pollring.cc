#include <nba/engines/knapp/micintrinsic.hh>
#include <nba/engines/knapp/pollring.hh>
#include <scif.h>
#include <cstring>
#include <cassert>

using namespace nba::knapp;

PollRing::PollRing(scif_epd_t epd, size_t len)
    : epd(epd), _len(len)
{
    assert(len > 0);
    alloc_bytes = ALIGN_CEIL(len * sizeof(poll_item_t), PAGE_SIZE);

    /* Clear the RMA area. */
    ring = (poll_item_t *) _mm_malloc(alloc_bytes, PAGE_SIZE);
    assert(nullptr != ring);
    volatile poll_item_t *_ring = ring;
    memset((void *) _ring, 0, alloc_bytes);
    ring_ra = (poll_item_t*) scif_register(epd, (void *) _ring,
                                           alloc_bytes, 0, SCIF_PROT_WRITE, 0);
    assert(ring_ra > 0);
}

PollRing::~PollRing()
{
    scif_unregister(epd, (off_t) ring, alloc_bytes);
}

void PollRing::get(poll_item_t &value)
{
}

void PollRing::put(const poll_item_t value)
{
}

void PollRing::wait(const unsigned idx, const poll_item_t value)
{
    poll_item_t volatile *_ring = ring;
    compiler_fence();
    while (_ring[idx] != value)
        insert_pause();
}

void PollRing::notify(const unsigned idx, const poll_item_t value)
{
    compiler_fence();
    ring[idx] = value;
}

void PollRing::remote_notify(const unsigned idx, const poll_item_t value)
{
    scif_fence_signal(epd, 0, 0,
                      (off_t) &ring_ra[idx],
                      (uint64_t) value,
                      SCIF_FENCE_INIT_SELF | SCIF_SIGNAL_REMOTE);
}

// vim: ts=8 sts=4 sw=4 et
