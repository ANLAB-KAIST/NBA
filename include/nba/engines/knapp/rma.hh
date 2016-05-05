#ifndef __NBA_KNAPP_RMA_HH__
#define __NBA_KNAPP_RMA_HH__

#include <cstdint>
#include <scif.h>

namespace nba { namespace knapp {

constexpr unsigned poll_itemsz = sizeof(uint64_t);

class RMABuffer
{
public:
    RMABuffer(scif_epd_t ep, size_t len, size_t elemsz);

    virtual ~RMABuffer();

    void *get_va(unsigned slot_id);
    void *get_ra(unsigned slot_id);

    void poll(unsigned slot_id);
    void write(unsigned slot_id);
    void signal(unsigned slot_id);

private:
    scif_epd_t ep;
    size_t _len;
    size_t elemsz;

    off_t base_addr;
};

}} //endns(nba::knapp)

#endif //__NBA_KNAPP_RMA_HH__

// vim: ts=8 sts=4 sw=4 et
