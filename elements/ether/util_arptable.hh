/*
 * arptable.{cc,hh} -- ARP resolver element
 * Eddie Kohler
 *
 * Copyright (c) 1999-2000 Massachusetts Institute of Technology
 * Copyright (c) 2005 Regents of the University of California
 * Copyright (c) 2008 Meraki, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, subject to the conditions
 * listed in the Click LICENSE file. These conditions include: you must
 * preserve this copyright notice, and you cannot mention the copyright
 * holders in advertising related to the Software without their permission.
 * The Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This
 * notice is a summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */

/* Moved from Click project & modified by Sangwook Ma on 13.09.14. */

#ifndef __NBA_UTIL_ARP_TABLE_HH__
#define __NBA_UTIL_ARP_TABLE_HH__


#include <rte_config.h>
#include <rte_rwlock.h>
#include <rte_cycles.h>
#include <rte_memory.h>
#include <rte_malloc.h>

#include <rte_common.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>

#include <rte_ether.h>


#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <list>
#include <vector>
#include <unordered_map>
#include <string>

#include <net/if_arp.h>
#include <netinet/ip.h>

using namespace std;

// Copied from <netinet/if_ether.h>
// If we include <if_ether.h>, it collides with rte_ether.h
struct  ether_arp {
    struct  arphdr ea_hdr;          /* fixed-size header */
    uint8_t arp_sha[6];             /* sender hardware address */
    uint8_t arp_spa[4];            /* sender protocol address */
    uint8_t arp_tha[6];             /* target hardware address */
    uint8_t arp_tpa[4];            /* target protocol address */
};

class EtherAddress {
public:
    // Mac address stored in network byte order.
    inline EtherAddress() {
        _data[0] = _data[1] = _data[2] = 0;
    }

    inline EtherAddress(struct ether_addr *addr) {
            _data[0] = addr->addr_bytes[0];
            _data[1] = addr->addr_bytes[1];
            _data[2] = addr->addr_bytes[2];
            _data[3] = addr->addr_bytes[3];
            _data[4] = addr->addr_bytes[4];
            _data[5] = addr->addr_bytes[5];
    }

    // From stackoverflow
    // http://stackoverflow.com/questions/7326123/convert-mac-address-stdstring-into-uint64-t
    // Input string in the form of "AA:BB:CC:DD:EE:FF"
    inline EtherAddress(string const& s) {
        uint8_t a[6];
        int last = -1;
        int rc = sscanf(s.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx%n",
                        a + 0, a + 1, a + 2, a + 3, a + 4, a + 5,
                        &last);
        if(rc != 6 || (int)s.size() != last) {
            fprintf(stderr, "NBA: EtherAddress::EtherAddress(std::string), wrong input format\n");
            exit(EXIT_FAILURE);
        }

        _data[0] = (a[0]);
        _data[1] = (a[1]);
        _data[2] = (a[2]);
        _data[3] = (a[3]);
        _data[4] = (a[4]);
        _data[5] = (a[5]);
    }

    ~EtherAddress() { }

    inline void set(uint8_t *addr) {
        int i;
        for (i=0; i<6; ++i) {
            _data[i] = addr[i];
        }
    }

    inline bool is_broadcast() {
        uint16_t sum = _data[0] + _data[1] + _data[2] + _data[3] + _data[4] + _data[5];
        return (sum == 0x5FA);
    }

    inline void set_broadcast() {
        _data[0] = _data[1] = _data[2] = _data[3] = _data[4] = _data[5] = 0xFF;
    }

    inline void put_to(struct ether_addr *addr) {
        addr->addr_bytes[0] = _data[0];
        addr->addr_bytes[1] = _data[1];
        addr->addr_bytes[2] = _data[2];
        addr->addr_bytes[3] = _data[3];
        addr->addr_bytes[4] = _data[4];
        addr->addr_bytes[5] = _data[5];
    }

    uint8_t _data[6];
};

// Converts ip address in uint32_t to uint8_t[4].
// from stackoverflow: http://stackoverflow.com/questions/6499183/converting-a-uint32-value-into-a-uint8-array4
void convert_ip_addr(uint32_t from, uint8_t *to);

class ARPTable { public:

    ARPTable();
    ~ARPTable();

    //const char *class_name() const        { return "ARPTable"; }

    int configure(uint32_t packet_capacity, uint32_t entry_capacity, uint32_t timeout);

    int lookup(uint32_t ip, EtherAddress *eth, uint32_t poll_timeout_j);
    EtherAddress lookup(uint32_t ip);
    //int insert(uint32_t ip, const EtherAddress &en, Packet **head = 0);
    int insert(uint32_t ip, const EtherAddress &en);
    void clear();
    void handle_timer();

    void slim(long now);

    rte_rwlock_t *get_rwlock() { return &_lock; };

    struct ARPEntry {
        uint32_t _ip;
        ARPEntry *_hashnext;
        EtherAddress _eth;
        long _input_time;
        //List<Packet*> _pkt_list;
        ARPEntry(uint32_t ip)
            : _ip(ip), _hashnext() {
            _eth.set_broadcast();
        }
        uint32_t hashkey() const {
            return _ip;
        }
    };

  private:
    rte_rwlock_t _lock;

    typedef unordered_map<uint32_t, ARPEntry*> EntryHashMap;
    typedef list<ARPEntry *> AgeList;
    EntryHashMap *_hash_map;
    AgeList *_age;
    uint32_t _entry_count;
    uint32_t _packet_count;
    uint32_t _entry_capacity;
    uint32_t _packet_capacity;
    long _timeout_j;

    ARPEntry *ensure(uint32_t ip, long now);
};

inline int
ARPTable::lookup(uint32_t ip, EtherAddress *eth, uint32_t poll_timeout_j)
{
    rte_rwlock_read_lock(&_lock);
    int r = 0;
    timeval tv;
    EntryHashMap::iterator it = _hash_map->find(ip);
    if (it != _hash_map->end())  {
        ARPEntry *entry = it->second;
        gettimeofday(&tv, NULL);
        long now = tv.tv_sec;
        if (entry->_input_time + _timeout_j > now) {
            *eth = entry->_eth;
            r = 1;
        }
    }
    rte_rwlock_read_unlock(&_lock);
    return r;
}

inline EtherAddress
ARPTable::lookup(uint32_t ip)
{
    EtherAddress eth;
    if (lookup(ip, &eth, 0) >= 0) {
        return eth;
    }
    else {
        eth.set_broadcast();
        return eth;
    }
}

#endif

// vim: ts=8 sts=4 sw=4 et
