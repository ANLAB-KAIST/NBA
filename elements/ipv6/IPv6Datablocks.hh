#ifndef __NBA_IPV6_DATABLOCKS_HH__
#define __NBA_IPV6_DATABLOCKS_HH__

#include "../../lib/packetbatch.hh"
#include "../../lib/datablock.hh"
#include "util_hash_table.hh"

namespace nba {

extern int dbid_ipv6_dest_addrs;
extern int dbid_ipv6_lookup_results;

class IPv6DestAddrsDataBlock : DataBlock
{
public:
    IPv6DestAddrsDataBlock() : DataBlock()
    {}

    virtual ~IPv6DestAddrsDataBlock()
    {}

    const char *name() const { return "ipv6.dest_addrs"; }

    void get_read_roi(struct read_roi_info *roi) const
    {
        // Dest IPv6 addr, whose format is in6_addr struct, is converted to uint128_t in preproc().
        roi->type = READ_PARTIAL_PACKET;
        roi->offset = 14 + 24;  /* offset of IPv6 destination address. */
        roi->length = sizeof(uint128_t);
        roi->align = 0;
    }

    void get_write_roi(struct write_roi_info *roi) const
    {
        roi->type = WRITE_NONE;
        roi->offset = 0;
        roi->length = 0;
        roi->align = 0;
    }
};

class IPv6LookupResultsDataBlock : DataBlock
{
public:
    IPv6LookupResultsDataBlock() : DataBlock()
    {}

    virtual ~IPv6LookupResultsDataBlock()
    {}

    const char *name() const { return "ipv6.lookup_results"; }

    void get_read_roi(struct read_roi_info *roi) const
    {
        roi->type = READ_NONE;
        roi->offset = 0;
        roi->length = 0;
        roi->align = 0;
    }

    void get_write_roi(struct write_roi_info *roi) const
    {
        roi->type = WRITE_FIXED_SEGMENTS;
        roi->offset = 0;
        roi->length = sizeof(uint16_t);
        roi->align = 0;
    }
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
