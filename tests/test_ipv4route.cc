#include <cstdint>
#include <cuda_runtime.h>
#include <nba/framework/datablock_shared.hh>
#include <gtest/gtest.h>
#include "../elements/ip/ip_route_core.hh"
#include "../elements/ip/IPlookup_kernel.hh"
#if 0
#require "../elements/ip/ip_route_core.o"
#require "../elements/ip/IPlookup_kernel.o"
#endif

using namespace std;
using namespace nba;

TEST(IPLookupTest, Loading) {
    ipv4route::route_hash_t tables[33];
    ipv4route::load_rib_from_file(tables, "configs/routing_info.txt");
    size_t num_entries = 0;
    for (int i = 0; i <= 32; i++) {
        printf("table[%d] size: %lu\n", i, tables[i].size());
        num_entries += tables[i].size();
    }
    EXPECT_EQ(282797, num_entries) << "All entries (lines) should exist.";

    // Add extra 32 entries and check overflowing.
    uint16_t *tbl24   = (uint16_t *) malloc(sizeof(uint16_t) * (ipv4route::get_TBL24_size() + 32));
    uint16_t *tbllong = (uint16_t *) malloc(sizeof(uint16_t) * (ipv4route::get_TBLlong_size() + 32));
    for (int i = 0; i < 32; i++)
        tbl24[ipv4route::get_TBL24_size() + i] = i;
    ipv4route::build_direct_fib(tables, tbl24, tbllong);
    for (int i = 0; i < 32; i++)
        EXPECT_EQ(i, tbl24[ipv4route::get_TBL24_size() + i]);
    free(tbl24);
    free(tbllong);
}


// vim: ts=8 sts=4 sw=4 et
