#include "util_hash_table.hh"
#include "util_routing_v6.hh"

using namespace std;
using namespace nba;

int RoutingTableV6::from_random(int seed, int count)
{
    srand(seed);
    for (int i = 0; i < count; i++) {
        int len = rand() % 128 + 1;
        uint128_t addr;
        uint16_t dest;
        addr.u32[0] = rand();
        addr.u32[1] = rand();
        addr.u32[2] = rand();
        addr.u32[3] = rand();
        dest = rand() % 65535 + 1;
        add(addr, len, dest);
    }
    return 0;
}

int RoutingTableV6::from_file(const char *)
{
    printf("Not implemented yet.");
    return 0;
}

void RoutingTableV6::add(uint128_t addr, int len, uint16_t dest)
{
    assert(len > 0 && len < 129);
    m_Tables[len-1]->insert(mask(addr,len), dest);
}

int RoutingTableV6::update(uint128_t, int, uint16_t)
{
    printf("Not implemented yet.");
    return 0;
}

int RoutingTableV6::remove(uint128_t, int)
{
    printf("Not implemented yet.");
    return 0;
}

int RoutingTableV6::build()
{
    build_lock_.acquire();
    if (m_IsBuilt) {
        build_lock_.release();
        return 0;
    }
    for (int i = 0; i < 128; i++){
        HashTable128 *table = m_Tables[i];
        int len = i;
        for (HashTable128::Iterator i = table->begin(); i != table->end(); ++i) {
            int start = 0;
            int end = 127;
            int len_marker = (start + end) / 2;
            while (len_marker != len  && start <= end) {
                uint128_t temp = mask(*i, len_marker + 1);
                uint16_t marker_dest = lookup(&temp);
                if (len_marker < len) {
                    m_Tables[len_marker]->insert(mask(*i, len_marker +1), marker_dest, IPV6_HASHTABLE_MARKER);
                }

                if (len < len_marker) {
                    end = len_marker - 1;
                } else if (len > len_marker) {
                    start = len_marker + 1;
                }

                len_marker = (start + end) / 2;
            }
        }
    }
    m_IsBuilt = true;
    build_lock_.release();
    return 0;
}

uint16_t RoutingTableV6::lookup(uint128_t *ip)
{
    // Note: lookup() method is also called from build().
    //       We should NOT place an assertion on m_IsBuilt here,
    //       and it should be done before calling this method
    //       elsewhere.

    int start = 0;
    int end = 127;
    uint16_t result = 0;
    do {
        int len = (start + end) / 2;

        uint16_t temp = m_Tables[len]->find(mask(*ip, len + 1));

        if (temp == 0) {
            end = len - 1;
        } else {
            result = temp;
            start = len + 1;
        }
    } while (start <= end);

    return result;
}

RoutingTableV6 *RoutingTableV6::clone()
{
    build_lock_.acquire();
    RoutingTableV6 *new_table = new RoutingTableV6();
    new_table->m_IsBuilt = m_IsBuilt;
    for (int i = 0; i < 128; i++)
        new_table->m_Tables[i]->clone_from(*m_Tables[i]);
    build_lock_.release();
    return new_table;
}

void RoutingTableV6::copy_to(RoutingTableV6 *new_table)
{
    build_lock_.acquire();

    if (new_table == NULL) {
        printf("NBA: RoutingTableV6:copy_to: argument not alloced.\n");
        exit(EXIT_FAILURE);
    }
    new_table->m_IsBuilt = m_IsBuilt;
    for (int i = 0; i < 128; i++)
        new_table->m_Tables[i]->clone_from(*m_Tables[i]);
    build_lock_.release();
    return;
}

// vim: ts=8 sts=4 sw=4 et
