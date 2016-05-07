#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/micintrinsic.hh>
#include <nba/engines/knapp/pollring.hh>
#include <scif.h>
#include <cstring>
#include <cassert>
#include <cstdio>

using namespace nba::knapp;

PollRing::PollRing(scif_epd_t epd, size_t len)
    : _epd(epd), _len(len), _peer_ra(-1)
{
    assert(len > 0);
    _alloc_bytes = ALIGN_CEIL(len * sizeof(poll_item_t), PAGE_SIZE);

    /* Clear the RMA area. */
    _local_va = (poll_item_t *) _mm_malloc(_alloc_bytes, PAGE_SIZE);
    assert(nullptr != _local_va);
    memset((void *) _local_va, 0, _alloc_bytes);
    off_t reg_result;
    reg_result = scif_register(_epd, (void *) _local_va,
                               _alloc_bytes, 0, SCIF_PROT_WRITE, 0);
    assert(SCIF_REGISTER_FAILED != reg_result);
    _local_ra = (poll_item_t *) reg_result;
}

PollRing::~PollRing()
{
    int rc;
    rc = scif_unregister(_epd, (off_t) _local_ra, _alloc_bytes);
    if (rc < 0)
        perror("~PollRing: scif_unregister");
    _mm_free(_local_va);
}

bool PollRing::wait(const unsigned idx, const poll_item_t value)
{
    poll_item_t volatile *_ring = _local_va;
    uint32_t count = 0;
    compiler_fence();
    while (_ring[idx] != value && count < KNAPP_SYNC_CYCLES) {
        insert_pause();
        count++;
    }
    /* Return true if waited too long. */
    return count != KNAPP_SYNC_CYCLES;
}

bool PollRing::poll(const unsigned idx, const poll_item_t value)
{
    poll_item_t volatile *_ring = _local_va;
    compiler_fence();
    return (_ring[idx] == value);
}

void PollRing::notify(const unsigned idx, const poll_item_t value)
{
    compiler_fence();
    _local_va[idx] = value;
}

void PollRing::remote_notify(const unsigned idx, const poll_item_t value)
{
    int rc;
    _local_va[idx] = value;
    rc = scif_fence_signal(_epd, 0, 0,
                           _peer_ra + sizeof(poll_item_t) * idx,
                           (uint64_t) value,
                           SCIF_FENCE_INIT_SELF | SCIF_SIGNAL_REMOTE);
    //assert(rc == 0);
}

// vim: ts=8 sts=4 sw=4 et
