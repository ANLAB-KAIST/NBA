#include <nba/core/intrinsic.hh>
#include <nba/engines/knapp/rma.hh>
#include <scif.h>
#include <rte_config.h>
#include <rte_malloc.h>
#include <cassert>
#include <cstdio>

using namespace nba::knapp;

RMABuffer::RMABuffer(
        scif_epd_t epd, size_t size, int node_id)
    : _epd(epd), _size(size), _extern_base(false)
{
    _local_va = (off_t) rte_malloc_socket("rma", size, PAGE_SIZE, node_id);
    assert(0 != _local_va);
    _local_ra = scif_register(_epd, (void *) _local_va, size, 0,
                            SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    assert(0 != _local_ra);
}

RMABuffer::RMABuffer(scif_epd_t epd, void *extern_arena, size_t size)
    : _epd(epd), _size(size), _extern_base(true)
{
    _local_va = (off_t) extern_arena;
    assert(0 != _local_va);
    _local_ra = scif_register(_epd, (void *) _local_va, size, 0,
                            SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    assert(0 != _local_ra);
}

RMABuffer::~RMABuffer()
{
    int rc;
    rc = scif_unregister(_epd, _local_ra, _size);
    if (rc < 0)
        perror("~RMABuffer: scif_unregister");
    if (!_extern_base)
        rte_free((void *) _local_va);
}

void RMABuffer::write(off_t offset, size_t size)
{
    int rc;
    assert(size <= _size);
    rc = scif_writeto(_epd, _local_ra + offset, size, _peer_ra + offset, 0);
    if (rc < 0)
        perror("RMABuffer: scif_writeto");
    assert(0 == rc);
}


// vim: ts=8 sts=4 sw=4 et
