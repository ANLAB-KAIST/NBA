#ifndef __NSHADER_IPv6_ROUTINGTABLE_HH__
#define __NSHADER_IPv6_ROUTINGTABLE_HH__

#include <cstdint>
#include "util_hash_table.hh"

#include "../../lib/types.hh"

using namespace std;
using namespace nshader;

inline uint128_t mask(const uint128_t aa, int len)
{
    len = 128 - len;
    uint128_t a = aa;
    assert(len >= 0 && len <= 128);

    if (len < 64) {
        a.u64[0] = ((a.u64[0]>>len)<<len);
    } else if (len < 128) {
        a.u64[1] = ((a.u64[1]>>(len-64))<<(len-64));
        a.u64[0] = 0;
    } else {
        a.u64[0] = 0;
        a.u64[1] = 0;
    }
    return a;
};

class RoutingTableV6
{
public:
    RoutingTableV6() : m_IsBuilt(false)
    {
        for (int i = 0; i < 128; i++) {
            // Currently all tables have the same DEFAULT_TABLE_SIZE;
            m_Tables[i] = new HashTable128();
        }
    }
    virtual ~RoutingTableV6()
    {
        for (int i = 0; i < 128; i++)
            delete m_Tables[i];
    }
    int from_random(int seed, int count);
    int from_file(const char* filename);
    void add(uint128_t addr, int len, uint16_t dest);
    int update(uint128_t addr, int len, uint16_t dest);
    int remove(uint128_t addr, int len);
    int build();
    uint16_t lookup(uint128_t *ip);
    RoutingTableV6 *clone();
    void copy_to(RoutingTableV6 *new_table);    // added function in modular-nshader

    HashTable128 *m_Tables[128];
    bool m_IsBuilt;
private:
    Lock build_lock_;
};

#endif

// vim: ts=8 sts=4 sw=4 et
