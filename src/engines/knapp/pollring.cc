#include <nba/core/intrinsic.hh>
#include <nba/engines/knapp/pollring.hh>
#include <cstring>
#include <cassert>
#include <rte_config.h>
#include <rte_ring.h>
#include <rte_malloc.h>
#include <scif.h>

using namespace nba::knapp;

static int _global_pollring_counter = 0;


PollRing::PollRing(scif_epd_t epd, size_t len, int node_id)
    : epd(epd), _len(len), node_id(node_id)
{
    int rc;
    char ringname[32];

    assert(len > 0);
    alloc_bytes = ALIGN_CEIL(len * sizeof(poll_item_t), PAGE_SIZE);

    /* Create a ring to keep track of item values. */
    snprintf(ringname, 32, "poll-id-pool-%d", _global_pollring_counter++);
    id_pool = rte_ring_create(ringname, len + 1, node_id, 0);
    assert(nullptr != id_pool);

    /* Fill the ring initially. */
    uintptr_t id_pool_content[len];
    for (unsigned i = 0; i < len; i++) {
        id_pool_content[i] = i;
    }
    rc = rte_ring_enqueue_bulk(id_pool, (void **) id_pool_content, len);
    assert(0 == rc);

    /* Clear the RMA area. */
    ring = (poll_item_t *) rte_malloc_socket("poll_ring",
                                             alloc_bytes,
                                             PAGE_SIZE, node_id);
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
    int rc;
    uintptr_t item;
    while (0 != (rc = rte_ring_dequeue(id_pool, (void **) &item))) {
        rte_pause();
    }
    value = (poll_item_t) item;
}

void PollRing::put(const poll_item_t value)
{
    rte_ring_enqueue(id_pool, (void *) value);
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
