#ifndef __NBA_KNAPP_RMA_HH__
#define __NBA_KNAPP_RMA_HH__

#include <cstdint>
#include <scif.h>

namespace nba { namespace knapp {

class RMABuffer
{
public:
#ifdef __MIC__
    RMABuffer(scif_epd_t epd, size_t size);
#else
    RMABuffer(scif_epd_t epd, size_t size, int node_id);
#endif
    RMABuffer(scif_epd_t epd, void *extern_arena, size_t size);

    virtual ~RMABuffer();

    void write(off_t offset, size_t size, bool sync = false);
    void read(off_t offset, size_t size, bool sync = false);

    off_t va() const { return _local_va; }
    off_t ra() const { return _local_ra; }
    void set_peer_ra(off_t value) { this->_peer_ra = value; }
    off_t peer_ra() const { return _peer_ra; }
#ifndef __MIC__
    void set_peer_va(off_t value) { this->_peer_va = value; }
    off_t peer_va() const { return _peer_va; }
#endif

private:
    scif_epd_t _epd;
    size_t _size;
    bool _extern_base;

    off_t _local_va;
    off_t _local_ra;
#ifndef __MIC__
    off_t _peer_va;
#endif
    off_t _peer_ra;
};

}} //endns(nba::knapp)

#endif //__NBA_KNAPP_RMA_HH__

// vim: ts=8 sts=4 sw=4 et
