#! /usr/bin/env python3
import nshader
import sys, os

for netdev in nshader.get_netdevices():
    print(netdev)
for coproc in nshader.get_coprocessors():
    print(coproc)
node_cpus = nshader.get_cpu_node_mapping()
for node_id, cpus in enumerate(node_cpus):
    print('Cores in NUMA node {0}: [{1}]'.format(node_id, ', '.join(map(str, cpus))))

system_params = {
    'IO_BATCH_SIZE': 64,
    'COMP_BATCH_SIZE': 64,
    'COPROC_PPDEPTH': 3 if nshader.emulate_io else 64,
}
print("# logical cores: {0}, # physical cores {1} (hyperthreading {2})".format(
    nshader.num_logical_cores, nshader.num_physical_cores,
    "enabled" if nshader.ht_enabled else "disabled"
))
_ht_diff = nshader.num_physical_cores if nshader.ht_enabled else 0

io_threads = [
    # core_id, list of (port_id, rxq_idx)
    nshader.IOThread(core_id=node_cpus[0][0], attached_rxqs=[(0, 0)], mode='normal'),
]

comp_threads = [
    # core_id
    nshader.CompThread(core_id=node_cpus[0][0] + _ht_diff),
]

coproc_threads = [
    # core_id, device_id
    nshader.CoprocThread(core_id=node_cpus[0][7] + _ht_diff, device_id=0),
]

queues = [
    nshader.Queue(node_id=0, template='swrx'),
    nshader.Queue(node_id=0, template='taskin'),
    nshader.Queue(node_id=0, template='taskout'),
]

thread_connections = [
    # from-thread, to-thread, queue-instance
    (io_threads[0], comp_threads[0], queues[0]),
    (comp_threads[0], coproc_threads[0], queues[1]),
    (coproc_threads[0], comp_threads[0], queues[2]),
]
