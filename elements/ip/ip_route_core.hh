#ifndef __NBA_IP_ROUTE_CORE_HH__
#define __NBA_IP_ROUTE_CORE_HH__

#include <cstdint>
#include <unordered_map>

#define TBL24_SIZE   ((1 << 24) + 1)
#define TBLLONG_SIZE ((1 << 24) + 1)

namespace nba {

namespace ipv4route {

typedef std::unordered_map<uint32_t, uint16_t> route_hash_t;

/* RIB/FIB management methods. */
extern int add_route(route_hash_t *tables, uint32_t addr,
                     uint16_t len, uint16_t nexthop);
extern int delete_route(route_hash_t *tables, uint32_t addr, uint16_t len);

/** Builds RIB from a set of IPv4 prefixes in a file. */
extern int load_rib_from_file(route_hash_t *tables, const char* filename);

/** Builds FIB from RIB, using DIR-24-8-BASIC scheme. */
extern int build_direct_fib(const route_hash_t *tables,
                            uint16_t *TBL24, uint16_t *TBLlong);

static inline int get_TBL24_size() { return TBL24_SIZE; }
static inline int get_TBLlong_size() { return TBLLONG_SIZE; }

/** The CPU version implementation. */
extern void direct_lookup(const uint16_t *TBL24, const uint16_t *TBLlong,
                          const uint32_t ip, uint16_t *dest);

} // endns(ipv4route)

} // endns(nba)

#endif

// vim: ts=8 sts=4 sw=4 et
