#! /usr/bin/env python
import gdb
import re

RTE_MAX_LCORE = 64

def rte_ring_count(val):
    prod_tail = int(val['prod']['tail'])
    cons_tail = int(val['cons']['tail'])
    prod_mask = int(val['prod']['mask'])
    return (prod_tail - cons_tail) & prod_mask

class FixedRingPrinter:
    type_regex = re.compile(r'^(nba::)?FixedRing<.*>$')

    def __init__(self, val):
        self.val = val

    def to_string(self):
        push_idx = int(self.val['push_idx'])
        pop_idx = int(self.val['pop_idx'])
        max_size = int(self.val['max_size'])
        size = int(self.val['count'])
        return 'FixedRing of length {0} with {1} items'.format(max_size, size)

    def children(self):
        push_idx = int(self.val['push_idx'])
        pop_idx = int(self.val['pop_idx'])
        max_size = int(self.val['max_size'])
        size = int(self.val['count'])
        for i in range(0, size):
            yield '', self.val['v_'][(pop_idx + i) % max_size]

    def display_hint(self):
        return 'array'

class DPDKRingPrinter:
    type_regex = re.compile(r'^rte_ring$')

    def __init__(self, val):
        self.val = val

    def to_string(self):
        name = self.val['name'].string()
        count = rte_ring_count(self.val)
        size = int(self.val['prod']['size'])
        return 'rte_ring "{name}" of length {0} with {1} queued items'.format(
            size, count, name=name,
        )

    def children(self):
        cons_head = int(self.val['cons']['head'])
        prod_tail = int(self.val['prod']['tail'])
        cons_size = int(self.val['cons']['size'])
        prod_mask = int(self.val['prod']['mask'])
        begin_idx = cons_head & prod_mask
        num_entries = prod_tail - cons_head
        for i in range(0, num_entries):
            yield 'void *', self.val['ring'][(begin_idx + i) % cons_size]

    def display_hint(self):
        return 'array'

class DPDKMempoolPrinter:
    type_regex = re.compile(r'^rte_mempool$')

    def __init__(self, val):
        self.val = val

    def to_string(self):
        name = self.val['name'].string()
        count = rte_ring_count(self.val['ring'])
        if int(self.val['cache_size']) > 0:
            for lcore_id in range(0, RTE_MAX_LCORE):
                count += int(self.val['local_cache'][lcore_id]['len'])
        size = int(self.val['size'])
        if count > size:
            count = size
        return 'rte_mempool "{name}" of size {0} with {1} available items'.format(
            size, count, name=name,
        )

avail_types = (FixedRingPrinter, DPDKRingPrinter, DPDKMempoolPrinter)

def lookup_type(val):
    lookup_tag = val.type.tag
    if lookup_tag is None:
        return None
    for t in avail_types:
        if t.type_regex.match(lookup_tag):
            return t(val)
    return None

gdb.pretty_printers.append(lookup_type)
print("NBA/DPDK pretty printer support is enabled.")
