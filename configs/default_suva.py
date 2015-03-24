#! /usr/bin/env python3
import nba

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
# - loadbalancer
# - loadbalancer_cpu_ratio (In LBs not using ratio, this value is ignored.)

system_params = {
    'IO_BATCH_SIZE': 32,
    'COMP_BATCH_SIZE': 32,
    'COMP_PPDEPTH': 32,
    'COPROC_PPDEPTH': 64,
}
print("# logical cores: {0}, # physical cores {1} (hyperthreading {2})".format(
    nba.num_logical_cores, nba.num_physical_cores,
    "enabled" if nba.ht_enabled else "disabled"
))
_ht_diff = nba.num_physical_cores if nba.ht_enabled else 0

# The following objects are not "real" -- just namedtuple instances.
# They only store metdata w/o actual side-effects such as creation of threads.

io_threads = [
    # core_id, list of (port_id, rxq_idx), mode
    nba.IOThread(core_id=node_cpus[0][0], attached_rxqs=[(0, 0)], mode='normal'),
    nba.IOThread(core_id=node_cpus[0][1], attached_rxqs=[(1, 0)], mode='normal'),
    nba.IOThread(core_id=node_cpus[1][0], attached_rxqs=[(2, 0)], mode='normal'),
    nba.IOThread(core_id=node_cpus[1][1], attached_rxqs=[(3, 0)], mode='normal'),
]
comp_threads = [
    # core_id
    nba.CompThread(core_id=node_cpus[0][0] + _ht_diff),
    nba.CompThread(core_id=node_cpus[0][1] + _ht_diff),
    nba.CompThread(core_id=node_cpus[1][0] + _ht_diff),
    nba.CompThread(core_id=node_cpus[1][1] + _ht_diff),
]

coproc_threads = [
    # core_id, device_id
    nba.CoprocThread(core_id=node_cpus[0][7], device_id=0),
    nba.CoprocThread(core_id=node_cpus[1][7], device_id=1),
]

comp_input_queues = [
    # node_id, template
    nba.Queue(node_id=0, template='swrx'),
    nba.Queue(node_id=0, template='swrx'),
    nba.Queue(node_id=1, template='swrx'),
    nba.Queue(node_id=1, template='swrx'),
]

coproc_input_queues = [
    # node_id, template
    nba.Queue(node_id=0, template='taskin'),
    nba.Queue(node_id=1, template='taskin'),
]

coproc_completion_queues = [
    # node_id, template
    nba.Queue(node_id=0, template='taskout'),
    nba.Queue(node_id=0, template='taskout'),
    nba.Queue(node_id=1, template='taskout'),
    nba.Queue(node_id=1, template='taskout'),
]

queues = comp_input_queues + coproc_input_queues + coproc_completion_queues

thread_connections = [
    # from-thread, to-thread, queue-instance
    (io_threads[0], comp_threads[0], comp_input_queues[0]),
    (io_threads[1], comp_threads[1], comp_input_queues[1]),
    (io_threads[2], comp_threads[2], comp_input_queues[2]),
    (io_threads[3], comp_threads[3], comp_input_queues[3]),
    (comp_threads[0], coproc_threads[0], coproc_input_queues[0]),
    (comp_threads[1], coproc_threads[0], coproc_input_queues[0]),
    (comp_threads[2], coproc_threads[1], coproc_input_queues[1]),
    (comp_threads[3], coproc_threads[1], coproc_input_queues[1]),
    (coproc_threads[0], comp_threads[0], coproc_completion_queues[0]),
    (coproc_threads[0], comp_threads[1], coproc_completion_queues[1]),
    (coproc_threads[1], comp_threads[2], coproc_completion_queues[2]),
    (coproc_threads[1], comp_threads[3], coproc_completion_queues[3]),
]

# cpu_ratio is only used in weighted random LBs and ignored in other ones.
LB_mode = str(os.environ.get('NBA_LOADBALANCER_MODE', 'CPUOnly'))
LB_cpu_ratio = float(os.environ.get('NBA_LOADBALANCER_CPU_RATIO', 1.0))
load_balancer = nba.LoadBalancer(mode=LB_mode, cpu_ratio=LB_cpu_ratio)
