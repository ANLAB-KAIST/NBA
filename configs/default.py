#! /usr/bin/env python3
# This configuration is for ANY system.

import nba, os
from pprint import pprint

netdevices = nba.get_netdevices()
for netdev in netdevices:
    print(netdev)
coprocessors = nba.get_coprocessors()
for coproc in coprocessors:
    print(coproc)
node_cpus = nba.get_cpu_node_mapping()
for node_id, cpus in enumerate(node_cpus):
    print('Cores in NUMA node {0}: [{1}]'.format(node_id, ', '.join(map(str, cpus))))

# The values read by the framework are:
# - system_params
# - io_threads
# - comp_threads
# - coproc_threads
# - queues
# - thread_connections

system_params = {
    'IO_BATCH_SIZE': int(os.environ.get('NBA_IO_BATCH_SIZE', 64)),
    'COMP_BATCH_SIZE': int(os.environ.get('NBA_COMP_BATCH_SIZE', 64)),
    'COPROC_PPDEPTH': int(os.environ.get('NBA_COPROC_PPDEPTH', 32)),
    'COPROC_CTX_PER_COMPTHREAD': 1,
}
print("IO batch size: {0[IO_BATCH_SIZE]}, computation batch size: {0[COMP_BATCH_SIZE]}".format(system_params))
print("Coprocessor pipeline depth: {0[COPROC_PPDEPTH]}".format(system_params))
print("# logical cores: {0}, # physical cores {1} (hyperthreading degree {2})".format(
    nba.num_logical_cores, nba.num_physical_cores, nba.ht_degree
))
_ht_diff = nba.num_physical_cores if nba.ht_enabled else 0
pmd = os.environ.get('NBA_PMD', 'ixgbe')

def leq_power_of_two(val):
    m = val if val & (val - 1) == 0 else (val >> 1)
    v = 1
    while v < m:
        v <<= 1
    return v

def clean_siblings(core_list):
    greater_siblings = set()
    new_core_list = []
    for c in core_list:
        if c in greater_siblings: continue
        sibling_list_path = '/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list'.format(c)
        with open(sibling_list_path, 'r', encoding='ascii') as f:
            siblings = map(int, f.read().strip().split(','))
            greater_siblings.update(s for s in siblings if s != c)
        new_core_list.append(c)
    return new_core_list

thread_connections = []

coproc_threads = []
coproc_input_queues = []
node_local_coprocs = []
for node_id, node_cores in enumerate(node_cpus):
    node_local_coprocs.append([coproc for coproc in coprocessors if coproc.numa_node == node_id])
    for node_local_idx, coproc in enumerate(node_local_coprocs[node_id]):
        core_id = node_cores[-(node_local_idx + 1)]
        coproc_threads.append(nba.CoprocThread(core_id=core_id + _ht_diff, device_id=coproc.device_id))
        coproc_input_queues.append(nba.Queue(node_id=node_id, template='taskin'))

io_threads = []
comp_threads = []
comp_input_queues = []
coproc_completion_queues = []
for node_id, node_cores in enumerate(node_cpus):
    node_local_netdevices = [netdev for netdev in netdevices if netdev.numa_node == node_id]
    num_coproc_in_node = len(node_local_coprocs[node_id])
    node_cores = clean_siblings(node_cores)
    if num_coproc_in_node > 0:
        io_cores_in_node = node_cores[:-num_coproc_in_node]
    else:
        io_cores_in_node = node_cores[:]
    if pmd in ('mlx4', 'mlnx_uio'):
        # The number of RXQs must be a power of two for Mellanox cards.
        io_cores_in_node = io_cores_in_node[:leq_power_of_two(len(io_cores_in_node))]
    for node_local_core_id, core_id in enumerate(io_cores_in_node):
        rxqs = [(netdev.device_id, node_local_core_id) for netdev in node_local_netdevices]
        io_threads.append(nba.IOThread(core_id=node_cpus[node_id][node_local_core_id], attached_rxqs=rxqs, mode='normal'))
        comp_threads.append(nba.CompThread(core_id=node_cpus[node_id][node_local_core_id] + _ht_diff))
        comp_input_queues.append(nba.Queue(node_id=node_id, template='swrx'))
        thread_connections.append((io_threads[-1], comp_threads[-1], comp_input_queues[-1]))
        if num_coproc_in_node > 0:
            coproc_completion_queues.append(nba.Queue(node_id=node_id, template='taskout'))

for coproc_thread in coproc_threads:
    node_id = nba.node_of_cpu(coproc_thread.core_id)
    node_local_comp_threads = [comp_thread for comp_thread in comp_threads
                               if nba.node_of_cpu(comp_thread.core_id) == node_id]
    for comp_thread in node_local_comp_threads:
        thread_connections.append((comp_thread, coproc_thread, coproc_input_queues[node_id]))
        thread_connections.append((coproc_thread, comp_thread, coproc_completion_queues[comp_threads.index(comp_thread)]))

pprint(io_threads)
pprint(coproc_threads)

queues = comp_input_queues + coproc_input_queues + coproc_completion_queues
