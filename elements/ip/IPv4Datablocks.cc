#include "IPv4Datablocks.hh"
#include <rte_malloc.h>

namespace nba {

int dbid_ipv4_dest_addrs;
int dbid_ipv4_lookup_results;

static DataBlock* db_ipv4_dest_addrs_ctor (void) {
    DataBlock *ptr = (DataBlock *) rte_malloc("datablock", sizeof(IPv4DestAddrsDataBlock), CACHE_LINE_SIZE);
    assert(ptr != nullptr);
    new (ptr) IPv4DestAddrsDataBlock();
    return ptr;
};
static DataBlock* db_ipv4_lookup_results_ctor (void) {
    DataBlock *ptr = (DataBlock *) rte_malloc("datablock", sizeof(IPv4LookupResultsDataBlock), CACHE_LINE_SIZE);
    assert(ptr != nullptr);
    new (ptr) IPv4LookupResultsDataBlock();
    return ptr;
};

declare_datablock("ipv4.dest_addrs", db_ipv4_dest_addrs_ctor, dbid_ipv4_dest_addrs);
declare_datablock("ipv4.lookup_results", db_ipv4_lookup_results_ctor, dbid_ipv4_lookup_results);

}

// vim: ts=8 sts=4 sw=4 et
