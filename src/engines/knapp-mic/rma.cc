#include <nba/engines/knapp/micintrinsic.hh>
#include <nba/engines/knapp/rma.hh>
#include <scif.h>
#include <cassert>
#include <cstdio>

using namespace nba::knapp;

RMABuffer::RMABuffer(
        scif_epd_t epd, size_t size)
    : _epd(epd), _size(size), _extern_base(false)
{
    _local_va = (off_t) _mm_malloc(size, PAGE_SIZE);
    assert(0 != _local_va);
    _local_ra = scif_register(epd, (void *) _local_va, size, 0,
                            SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    assert(SCIF_REGISTER_FAILED != _local_ra);
}

RMABuffer::RMABuffer(scif_epd_t epd, void *extern_arena, size_t size)
    : _epd(epd), _size(size), _extern_base(true)
{
    _local_va = (off_t) extern_arena;
    assert(0 != _local_va);
    _local_ra = scif_register(_epd, (void *) _local_va, size, 0,
                            SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    assert(SCIF_REGISTER_FAILED != _local_ra);
}

RMABuffer::~RMABuffer()
{
    int rc;
    rc = scif_unregister(_epd, _local_ra, _size);
    if (rc < 0)
        perror("~RMABuffer: scif_unregister");
    if (!_extern_base)
        _mm_free((void *) _local_va);
}

void RMABuffer::write(off_t offset, size_t size, bool sync)
{
    int rc;
    assert(size <= _size);
    rc = scif_writeto(_epd, _local_ra + offset, size, _peer_ra + offset, 0);
    assert(0 == rc);
    if (sync) {
        int mark;
        rc = scif_fence_mark(_epd, SCIF_FENCE_INIT_SELF, &mark);
        assert(0 == rc);
        rc = scif_fence_wait(_epd, mark);
        assert(0 == rc);
    }
}

void RMABuffer::read(off_t offset, size_t size, bool sync)
{
    int rc;
    assert(size <= _size);
    rc = scif_readfrom(_epd, _local_ra + offset, size, _peer_ra + offset, 0);
    if (rc < 0)
        perror("RMABuffer: scif_readfrom");
    assert(0 == rc);
    if (sync) {
        int mark;
        rc = scif_fence_mark(_epd, SCIF_FENCE_INIT_SELF, &mark);
        assert(0 == rc);
        rc = scif_fence_wait(_epd, mark);
        assert(0 == rc);
    }
}


// vim: ts=8 sts=4 sw=4 et
