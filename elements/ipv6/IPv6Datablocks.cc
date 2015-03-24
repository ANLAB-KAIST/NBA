#include "IPv6Datablocks.hh"
#include <rte_malloc.h>

namespace nshader {

int dbid_ipv6_dest_addrs;
int dbid_ipv6_lookup_results;

static DataBlock* db_ipv6_dest_addrs_ctor (void) {
    DataBlock *ptr = (DataBlock *) rte_malloc("datablock", sizeof(IPv6DestAddrsDataBlock), CACHE_LINE_SIZE);
    assert(ptr != nullptr);
    new (ptr) IPv6DestAddrsDataBlock();
    return ptr;
};
static DataBlock* db_ipv6_lookup_results_ctor (void) {
    DataBlock *ptr = (DataBlock *) rte_malloc("datablock", sizeof(IPv6LookupResultsDataBlock), CACHE_LINE_SIZE);
    assert(ptr != nullptr);
    new (ptr) IPv6LookupResultsDataBlock();
    return ptr;
};

declare_datablock("ipv6.dest_addrs", db_ipv6_dest_addrs_ctor, dbid_ipv6_dest_addrs);
declare_datablock("ipv6.lookup_resutls", db_ipv6_lookup_results_ctor, dbid_ipv6_lookup_results);

}

// vim: ts=8 sts=4 sw=4 et
