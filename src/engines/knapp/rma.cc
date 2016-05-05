#include <nba/engines/knapp/rma.hh>
#include <scif.h>

using namespace nba::knapp;

RMABuffer::RMABuffer(
        scif_epd_t ep, size_t len, size_t elemsz)
    : ep(ep), _len(len), elemsz(elemsz)
{
    //scif_register();
}

RMABuffer::~RMABuffer()
{
    //scif_unregister();
}

void RMABuffer::write(unsigned slot_id)
{
    //scif_writeto(ep, loffset, len, roffset, 0);
}

void RMABuffer::signal(unsigned slot_id)
{
    //scif_fence_signal(ep, 0, 0, poll_roffset, poll_itemsz);
}


// vim: ts=8 sts=4 sw=4 et
