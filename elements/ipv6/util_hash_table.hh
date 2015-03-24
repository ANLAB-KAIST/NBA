#ifndef __NSHADER_UTIL_HASH_TABLE_HH__
#define __NSHADER_UTIL_HASH_TABLE_HH__

#include <stdint.h>
#include <unistd.h>

#define IPV6_DEFAULT_HASHTABLE_SIZE 65536
#define IPV6_HASHTABLE_MARKER 0x0002
#define IPV6_HASHTABLE_EMPTY  0x0000
#define IPV6_HASHTABLE_PREFIX 0x0001

// A 128-bit unsigned integer.

union uint128_t {
        uint32_t u32[4];
        uint64_t u64[2];
    uint8_t  u8[16];

    void set_ignored() {
        u64[0] = 0xffffffffffffffffu;
        u64[1] = 0xffffffffffffffffu;
    }

    bool is_ignored() {
        return u64[0] == 0xffffffffffffffffu && u64[0] == 0xffffffffffffffffu;
    }
};

inline bool operator == (const uint128_t &key1, const uint128_t &key2) {
        return key1.u64[0] == key2.u64[0] && key1.u64[1] == key2.u64[1];
}
inline bool operator != (const uint128_t &key1, const uint128_t &key2) {
        return key1.u64[0] != key2.u64[0] || key1.u64[1] != key2.u64[1];
}

// An item in the hash table.

struct Item {
    uint128_t key;
    uint16_t val;
    uint16_t state;
    uint32_t next;
};

class HashTable128
{
public:
    HashTable128(int tablesize = IPV6_DEFAULT_HASHTABLE_SIZE);
    ~HashTable128();
    int insert(uint128_t key, uint16_t val, uint16_t state = IPV6_HASHTABLE_PREFIX);
    uint32_t find(uint128_t key);
    void clone_from(HashTable128 &table);

public:

    //iterates non marker
    class Iterator
    {
    private:
        Item* m_Table;
        int m_TableSize;
        int m_CurrentIndex;
    public:
        Iterator(Item* Table, int TableSize ,int index = 0)
            : m_Table(Table), m_TableSize(TableSize), m_CurrentIndex(index)
        {
            while(!( m_Table[m_CurrentIndex].state & IPV6_HASHTABLE_PREFIX) && m_CurrentIndex < m_TableSize)
                m_CurrentIndex++;
        }
        Iterator& operator++ ()
        {
            if(m_Table[m_CurrentIndex].state & IPV6_HASHTABLE_PREFIX)
                m_CurrentIndex++;

            while(m_CurrentIndex < m_TableSize){
                if (m_Table[m_CurrentIndex].state == IPV6_HASHTABLE_PREFIX)
                    break;
                m_CurrentIndex++;
            }
            while(m_CurrentIndex >= m_TableSize &&  m_CurrentIndex < 2 * m_TableSize) {
                if ( !(m_Table[m_CurrentIndex].state & IPV6_HASHTABLE_MARKER) )
                    break;
                m_CurrentIndex++;
            }
            return *this;
        }
        uint128_t& operator* ()
        {
            return m_Table[m_CurrentIndex].key;
        }
        bool operator !=(const Iterator& b){
            return (m_CurrentIndex != b.m_CurrentIndex);
        }
    };
    Iterator begin() { return Iterator(m_Table, m_TableSize, 0);}
    Iterator end() { return Iterator(m_Table, m_TableSize, m_NextChain);}

    int m_TableSize;
    int m_NextChain;
    Item *m_Table;
};

#endif

// vim: ts=8 sts=4 sw=4 et
