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
    off_t reg_result;
    reg_result = scif_register(epd, (void *) _ring,
                               alloc_bytes, 0, SCIF_PROT_WRITE, 0);
    assert(SCIF_REGISTER_FAILED != reg_result);
    ring_ra = (poll_item_t *) reg_result;
}

PollRing::~PollRing()
{
    int rc;
    rc = scif_unregister(epd, (off_t) ring_ra, alloc_bytes);
    assert(0 == rc);
    _mm_free(ring);
}

void PollRing::wait(const unsigned idx, const poll_item_t value)
{
    poll_item_t volatile *_ring = ring;
    compiler_fence();
    while (_ring[idx] != value)
        insert_pause();
}

bool PollRing::poll(const unsigned idx, const poll_item_t value)
{
    poll_item_t volatile *_ring = ring;
    compiler_fence();
    return (_ring[idx] == value);
}

void PollRing::notify(const unsigned idx, const poll_item_t value)
{
    compiler_fence();
    ring[idx] = value;
}

void PollRing::remote_notify(const unsigned idx, const poll_item_t value)
{
    int rc;
    ring[idx] = value;
    rc = scif_fence_signal(epd, 0, 0,
                           (off_t) &ring_ra[idx],
                           (uint64_t) value,
                           SCIF_FENCE_INIT_SELF | SCIF_SIGNAL_REMOTE);
    assert(rc == 0);
}

// vim: ts=8 sts=4 sw=4 et
