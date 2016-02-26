#! /usr/bin/env python3
import nba
import sys, os

for netdev in nba.get_netdevices():
    print(netdev)
coprocessors = nba.get_coprocessors()
for coproc in coprocessors:
    print(coproc)
node_cpus = nba.get_cpu_node_mapping()
for node_id, cpus in enumerate(node_cpus):
    print('Cores in NUMA node {0}: [{1}]'.format(node_id, ', '.join(map(str, cpus))))

system_params = {
    'IO_BATCH_SIZE': 64,
    'COMP_BATCH_SIZE': 64,
    'COPROC_PPDEPTH': 3 if nba.emulate_io else 32,
}
print("# logical cores: {0}, # physical cores {1} (hyperthreading {2})".format(
    nba.num_logical_cores, nba.num_physical_cores,
    "enabled" if nba.ht_enabled else "disabled"
))
_ht_diff = nba.num_physical_cores if nba.ht_enabled else 0

io_threads = [
    # core_id, list of (port_id, rxq_idx)
    nba.IOThread(core_id=node_cpus[0][0], attached_rxqs=[(0, 0)], mode='normal'),
]

comp_threads = [
    # core_id
    nba.CompThread(core_id=node_cpus[0][0] + _ht_diff),
]

coproc_threads = [
    # core_id, device_id
    nba.CoprocThread(core_id=node_cpus[0][7] + _ht_diff, device_id=coprocessors[0].device_id),
]

queues = [
    nba.Queue(node_id=0, template='swrx'),
    nba.Queue(node_id=0, template='taskin'),
    nba.Queue(node_id=0, template='taskout'),
]

thread_connections = [
    # from-thread, to-thread, queue-instance
    (io_threads[0], comp_threads[0], queues[0]),
    (comp_threads[0], coproc_threads[0], queues[1]),
    (coproc_threads[0], comp_threads[0], queues[2]),
]
