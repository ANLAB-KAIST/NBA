#include <nba/core/intrinsic.hh>
#include <nba/engines/knapp/rma.hh>
#include <scif.h>
#include <rte_config.h>
#include <rte_malloc.h>
#include <cassert>

using namespace nba::knapp;

RMABuffer::RMABuffer(
        scif_epd_t epd, size_t sz, int node_id)
    : epd(epd), size(sz), extern_base(false)
{
    va_base = (off_t) rte_malloc_socket("rma", sz, PAGE_SIZE, node_id);
    assert(0 != va_base);
    ra_base = scif_register(epd, (void *) va_base, sz, 0,
                            SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    assert(0 != ra_base);
}

RMABuffer::RMABuffer(scif_epd_t epd, void *extern_arena, size_t sz)
    : epd(epd), size(sz), extern_base(true)
{
    va_base = (off_t) extern_arena;
    assert(0 != va_base);
    ra_base = scif_register(epd, (void *) va_base, sz, 0,
                            SCIF_PROT_READ | SCIF_PROT_WRITE, 0);
    assert(0 != ra_base);
}

RMABuffer::~RMABuffer()
{
    int rc;
    rc = scif_unregister(epd, ra_base, size);
    assert(0 == rc);
    if (!extern_base)
        rte_free((void *) va_base);
}

void RMABuffer::write(off_t offset, size_t len)
{
    scif_writeto(epd, va_base + offset, len, ra_base + offset, 0);
}


// vim: ts=8 sts=4 sw=4 et
