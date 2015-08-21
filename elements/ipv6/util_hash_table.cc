#include <cassert>
#include <cstring>
#include <cstdio>

#include <unistd.h>

#include "util_hash_table.hh"

typedef uint32_t u32;
typedef uint8_t u8;
#include "util_jhash.h"

using namespace std;
using namespace nba;

static void print128(uint128_t a)
{
    for (int i = 0; i < 64; i++) {
        printf("%ld", (a.u64[1] >> (63 - i) & 0x1));
    }
    for (int i = 0; i < 64; i++) {
        printf("%ld", (a.u64[0] >> (63 - i) & 0x1));
    }
}

#define HASH(x, y) (jhash2((u32*)&x, 4, 0) % y)

HashTable128::HashTable128(int tablesize)
{
    m_TableSize = tablesize;
    m_Table = new Item[m_TableSize * 2]; //allocate double space. bottom half will be used for chaining
    memset(m_Table, 0, sizeof(Item) * m_TableSize * 2);
    m_NextChain = m_TableSize;
}

HashTable128::~HashTable128()
{
    if (m_Table)
        delete m_Table;
}

int HashTable128::insert(uint128_t key, uint16_t val, uint16_t state)
{
    uint32_t index = HASH(key, m_TableSize);
    int ret = 0;

    //if hash key collision exist
    if (m_Table[index].state != IPV6_HASHTABLE_EMPTY) {
        while (m_Table[index].key != key) {
            if (m_Table[index].next == 0) {
                assert(m_NextChain < m_TableSize * 2 - 1);
                m_Table[index].next = m_NextChain;
                m_Table[m_NextChain].key = key;
                m_NextChain++;
            }
            index = m_Table[index].next;
        }
    }

    m_Table[index].key = key;
    m_Table[index].val = val;
    m_Table[index].state |= state;
    m_Table[index].next = 0;

    return ret;
}

uint32_t HashTable128::find(uint128_t key)
{
    uint32_t index = HASH(key, m_TableSize);
    uint16_t buf[2] = {0,0};
    uint32_t *ret = (uint32_t*)&buf;
    if (m_Table[index].state != IPV6_HASHTABLE_EMPTY) {
        if (m_Table[index].key == key){
            buf[0] = m_Table[index].val;
            buf[1] = m_Table[index].state;
        } else {

            index = m_Table[index].next;
            while (index != 0) {
                if (m_Table[index].key == key){
                    buf[0] = m_Table[index].val;
                    buf[1] = m_Table[index].state;
                    break;
                }
                index = m_Table[index].next;
            }
        }
    }
    return *ret;
}

void HashTable128::clone_from(HashTable128 &table)
{
    assert(m_TableSize >= table.m_TableSize);
    m_TableSize = table.m_TableSize;
    m_NextChain = table.m_NextChain;
    memcpy(m_Table, table.m_Table, sizeof(Item) * m_TableSize * 2);
}

// vim: ts=8 sts=4 sw=4 et
