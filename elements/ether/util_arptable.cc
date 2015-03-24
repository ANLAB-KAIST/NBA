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

#include "util_arptable.hh"

void convert_ip_addr(uint32_t from, uint8_t *to) {
    to[0] = (from & 0x000000ff);
    to[1] = (from & 0x0000ff00) >> 8;
    to[2] = (from & 0x00ff0000) >> 16;
    to[3] = (from & 0xff000000) >> 24;
}

ARPTable::ARPTable()
    : _entry_capacity(0), _packet_capacity(2048)//, _expire_timer(this)
{
    _entry_count = _packet_count = 0;
    rte_rwlock_init(&_lock);

    _hash_map = new EntryHashMap();
    _age = new AgeList();
}

ARPTable::~ARPTable()
{
    clear();
    delete _hash_map;
    delete _age;
}

int ARPTable::configure(uint32_t packet_capacity, uint32_t entry_capacity, uint32_t timeout)
{
    _packet_capacity = packet_capacity;
    _entry_capacity = entry_capacity;
    _timeout_j = (long)timeout;

    return 0;
}

void ARPTable::clear()
{
    // Walk the arp cache table and free any stored packets and arp entries.
    ARPEntry *entry_ptr;
    for (EntryHashMap::iterator it = _hash_map->begin(); it != _hash_map->end(); ++it) {
        entry_ptr = it->second;
        _hash_map->erase(it);
        delete entry_ptr;
        //ARPEntry *ae = _hash_map.erase(it);
        //while (Packet *p = ae->_head) {
        //  ae->_head = p->next();
        //    p->kill();
        //    ++_drops;
        //}
    }
    _entry_count = _packet_count = 0;
}

void ARPTable::slim(long now)
{
    ARPEntry *ae, *entry_ptr;
    EntryHashMap::iterator map_it;
    AgeList::iterator list_it;
    AgeList::iterator list_it_to_erase;

    // Delete old entries.
    //while ((ae = _age->front())
    //   && ( (entry->_input_time + _timeout_j < now)
    //     || (_entry_capacity && _entry_count > _entry_capacity))) {

    list_it = _age->begin();
    while (list_it != _age->end()) {
        ae = *list_it;
        if (ae->_input_time + _timeout_j < now) {
            map_it = _hash_map->find(ae->_ip);
            entry_ptr = map_it->second;
            _hash_map->erase(map_it);


            // while (Packet *p = ae->_head) {
            //  ae->_head = p->next();
            //  p->kill();
            //  --_packet_count;
            //  ++_drops;
            //}


            --_entry_count;
            list_it_to_erase = list_it;
            ++list_it;

            _age->erase(list_it_to_erase);
            delete entry_ptr;
        }
    }

    /*
    ARPEntry *ae;

    // Delete old entries.
    while ((ae = _age.front())
       && (ae->expired(now, _timeout_j)
           || (_entry_capacity && _entry_count > _entry_capacity))) {
    _hash_map.erase(ae->_ip);
    _age.pop_front();

    while (Packet *p = ae->_head) {
        ae->_head = p->next();
        p->kill();
        --_packet_count;
        ++_drops;
    }

    _alloc.deallocate(ae);
    --_entry_count;
    }

    // Mark entries for polling, and delete packets to make space.
    while (_packet_capacity && _packet_count > _packet_capacity) {
    while (ae->_head && _packet_count > _packet_capacity) {
        Packet *p = ae->_head;
        if (!(ae->_head = p->next()))
        ae->_tail = 0;
        p->kill();
        --_packet_count;
        ++_drops;
    }
    ae = ae->_age_link.next();
    }
    */
}

void ARPTable::handle_timer()
{
    // Expire any old entries, and make sure there's room for at least one packet.
    struct timeval tv;
    gettimeofday(&tv, NULL);
    rte_rwlock_write_lock(this->get_rwlock());
    this->slim(tv.tv_sec);
    rte_rwlock_write_unlock(this->get_rwlock());
}

ARPTable::ARPEntry * ARPTable::ensure(uint32_t ip, long now)
{
    rte_rwlock_write_lock(&_lock);
    EntryHashMap::iterator it = _hash_map->find(ip);
    if (it != _hash_map->end()) {
        ++_entry_count;
        if (_entry_capacity && _entry_count > _entry_capacity)
            slim(now);

        ARPEntry *ae = new ARPEntry(ip);
        ae->_input_time = now;
        //_hash_map.set(it, ae);
        _hash_map->insert(EntryHashMap::value_type(ip, ae));

        _age->push_back(ae);
    }
    return it->second;
}

int ARPTable::insert(uint32_t ip, const EtherAddress &eth)//, Packet **head)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    ARPEntry *ae = ensure(ip, tv.tv_sec);
    if (!ae)
        return -ENOMEM;

    ae->_eth = eth;
    ae->_input_time = tv.tv_sec;

    rte_rwlock_write_unlock(&_lock);
    return 0;
}

//EXPORT_ELEMENT(ARPTable)  // Sangwook: include as element?

// vim: ts=8 sts=4 sw=4 et
