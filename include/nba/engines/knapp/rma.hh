#ifndef __NBA_KNAPP_RMA_HH__
#define __NBA_KNAPP_RMA_HH__

#include <cstdint>
#include <scif.h>

namespace nba { namespace knapp {

class RMABuffer
{
public:
#ifdef __MIC__
    RMABuffer(scif_epd_t epd, size_t sz);
#else
    RMABuffer(scif_epd_t epd, size_t sz, int node_id);
#endif
    RMABuffer(scif_epd_t epd, void *extern_arena, size_t sz);

    virtual ~RMABuffer();

    void write(off_t offset, size_t len);

    off_t get_va() const { return va_base; }
    off_t get_ra() const { return ra_base; }

private:
    scif_epd_t epd;
    size_t size;
    bool extern_base;

    off_t va_base;
    off_t ra_base;
};

}} //endns(nba::knapp)

#endif //__NBA_KNAPP_RMA_HH__

// vim: ts=8 sts=4 sw=4 et
