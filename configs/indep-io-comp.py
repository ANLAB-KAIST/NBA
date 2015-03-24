#! /usr/bin/env python3
import nba, os
import sys

for netdev in nba.get_netdevices():
    print(netdev)
for coproc in nba.get_coprocessors():
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
}
print("# logical cores: {0}, # physical cores {1} (hyperthreading {2})".format(
    nba.num_logical_cores, nba.num_physical_cores,
    "enabled" if nba.ht_enabled else "disabled"
))
_ht_diff = nba.num_physical_cores if nba.ht_enabled else 0

io_threads = [
    # core_id, list of (port_id, rxq_idx)
    nba.IOThread(core_id=node_cpus[0][0], attached_rxqs=[(0, 0), (1, 0), (2, 0), (3, 0)], mode='normal'),
    nba.IOThread(core_id=node_cpus[0][1], attached_rxqs=[(0, 1), (1, 1), (2, 1), (3, 1)], mode='normal'),
    nba.IOThread(core_id=node_cpus[0][2], attached_rxqs=[(0, 2), (1, 2), (2, 2), (3, 2)], mode='normal'),
    nba.IOThread(core_id=node_cpus[0][3], attached_rxqs=[(0, 3), (1, 3), (2, 3), (3, 3)], mode='normal'),
    nba.IOThread(core_id=node_cpus[1][0], attached_rxqs=[(4, 0), (5, 0), (6, 0), (7, 0)], mode='normal'),
    nba.IOThread(core_id=node_cpus[1][1], attached_rxqs=[(4, 1), (5, 1), (6, 1), (7, 1)], mode='normal'),
    nba.IOThread(core_id=node_cpus[1][2], attached_rxqs=[(4, 2), (5, 2), (6, 2), (7, 2)], mode='normal'),
    nba.IOThread(core_id=node_cpus[1][3], attached_rxqs=[(4, 3), (5, 3), (6, 3), (7, 3)], mode='normal'),
]
comp_threads = [
    # core_id
    nba.CompThread(core_id=node_cpus[0][4]),
    nba.CompThread(core_id=node_cpus[0][5]),
    nba.CompThread(core_id=node_cpus[0][6]),
    nba.CompThread(core_id=node_cpus[0][7]),
    nba.CompThread(core_id=node_cpus[1][4]),
    nba.CompThread(core_id=node_cpus[1][5]),
    nba.CompThread(core_id=node_cpus[1][6]),
    nba.CompThread(core_id=node_cpus[1][7]),
]

coproc_threads = [
    # core_id, device_id
    #nba.CoprocThread(core_id=node_cpus[0][7] + _ht_diff, device_id=0),
    #nba.CoprocThread(core_id=node_cpus[1][7] + _ht_diff, device_id=1),
]

comp_input_queues = [
    # node_id, template
    nba.Queue(node_id=0, template='swrx'),
    nba.Queue(node_id=0, template='swrx'),
    nba.Queue(node_id=0, template='swrx'),
    nba.Queue(node_id=0, template='swrx'),
    nba.Queue(node_id=1, template='swrx'),
    nba.Queue(node_id=1, template='swrx'),
    nba.Queue(node_id=1, template='swrx'),
    nba.Queue(node_id=1, template='swrx'),
]

coproc_input_queues = [
    # node_id, template
    #nba.Queue(node_id=0, template='taskin'),
    #nba.Queue(node_id=1, template='taskin'),
]

coproc_completion_queues = [
    # node_id, template
    #nba.Queue(node_id=0, template='taskout'),
    #nba.Queue(node_id=1, template='taskout'),
]

queues = comp_input_queues + coproc_input_queues + coproc_completion_queues

thread_connections = [
    # from-thread, to-thread, queue-instance
    (io_threads[0], comp_threads[0], comp_input_queues[0]),
    (io_threads[1], comp_threads[1], comp_input_queues[1]),
    (io_threads[2], comp_threads[2], comp_input_queues[2]),
    (io_threads[3], comp_threads[3], comp_input_queues[3]),
    (io_threads[4], comp_threads[4], comp_input_queues[4]),
    (io_threads[5], comp_threads[5], comp_input_queues[5]),
    (io_threads[6], comp_threads[6], comp_input_queues[6]),
    (io_threads[7], comp_threads[7], comp_input_queues[7]),
]

LB_MODE = str(os.environ.get('NBA_LOADBALANCER_MODE', 'CPUOnlyLB'))
LB_CPU_RATIO = float(os.environ.get('NBA_LOADBALANCER_CPU_RATIO', 1.0))
load_balancer = nba.LoadBalancer(mode=LB_MODE, cpu_ratio=LB_CPU_RATIO)
