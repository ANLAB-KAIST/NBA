#ifndef __NSHADER_IPV4_DATABLOCKS_HH__
#define __NSHADER_IPV4_DATABLOCKS_HH__

#include "../../lib/packetbatch.hh"
#include "../../lib/datablock.hh"

namespace nshader {

extern int dbid_ipv4_dest_addrs;
extern int dbid_ipv4_lookup_results;

class IPv4DestAddrsDataBlock : DataBlock
{
public:
    IPv4DestAddrsDataBlock() : DataBlock()
    {}

    virtual ~IPv4DestAddrsDataBlock()
    {}

    const char *name() const { return "ipv4.dest_addrs"; }

    void get_read_roi(struct read_roi_info *roi) const
    {
        roi->type = READ_PARTIAL_PACKET;
        roi->offset = 14 + 16;  /* offset of IPv4 destination address */
        roi->length = 4;
        roi->align = 0;
    }

    void get_write_roi(struct write_roi_info *roi) const
    {
        roi->type = WRITE_NONE;
        roi->offset = 0;
        roi->length = 0;
    }
};

class IPv4LookupResultsDataBlock : DataBlock
{
public:
    IPv4LookupResultsDataBlock() : DataBlock()
    {}

    virtual ~IPv4LookupResultsDataBlock()
    {}

    const char *name() const { return "ipv4.lookup_results"; }

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
