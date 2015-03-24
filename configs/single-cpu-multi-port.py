#! /usr/bin/env python3
import nshader, os

for netdev in nshader.get_netdevices():
    print(netdev)
for coproc in nshader.get_coprocessors():
    print(coproc)
node_cpus = nshader.get_cpu_node_mapping()
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
    'IO_BATCH_SIZE': int(os.environ.get('NSHADER_IO_BATCH_SIZE', 32)),
    'COMP_BATCH_SIZE': int(os.environ.get('NSHADER_COMP_BATCH_SIZE', 32)),
    'COMP_PPDEPTH': int(os.environ.get('NSHADER_COMP_PPDEPTH', 16)),
    'COPROC_PPDEPTH': int(os.environ.get('NSHADER_COPROC_PPDEPTH', 64)),
}
print("IO batch size: {0[IO_BATCH_SIZE]}, computation batch size: {0[COMP_BATCH_SIZE]}".format(system_params))
print("Computation pipeline depth: {0[COMP_PPDEPTH]}".format(system_params))
print("Coprocessor pipeline depth: {0[COPROC_PPDEPTH]}".format(system_params))
print("# logical cores: {0}, # physical cores {1} (hyperthreading {2})".format(
    nshader.num_logical_cores, nshader.num_physical_cores,
    "enabled" if nshader.ht_enabled else "disabled"
))
_ht_diff = nshader.num_physical_cores if nshader.ht_enabled else 0

# The following objects are not "real" -- just namedtuple instances.
# They only store metdata w/o actual side-effects such as creation of threads.


no_port = int(os.environ.get('NSHADER_SINGLE_CPU_MULTI_PORT', 1))

print ("using " + str(no_port) + " ports for 1 cpu")

attached_rxqs_temp = []
for i in range(no_port):
    attached_rxqs_temp.append((i, 0))

io_threads = [
    # core_id, list of (port_id, rxq_idx), mode
    nshader.IOThread(core_id=node_cpus[0][0], attached_rxqs=attached_rxqs_temp, mode='normal'),
]
comp_threads = [
    # core_id
    nshader.CompThread(core_id=node_cpus[0][0] + _ht_diff),
]

coproc_threads = [
    # core_id, device_id
    nshader.CoprocThread(core_id=node_cpus[0][7] + _ht_diff, device_id=0),
]

comp_input_queues = [
    # node_id, template
    nshader.Queue(node_id=0, template='swrx'),
]

coproc_input_queues = [
    # node_id, template
    nshader.Queue(node_id=0, template='taskin'),
]

coproc_completion_queues = [
    # node_id, template
    nshader.Queue(node_id=0, template='taskout'),
]

queues = comp_input_queues + coproc_input_queues + coproc_completion_queues

thread_connections = [
    # from-thread, to-thread, queue-instance
    (io_threads[0], comp_threads[0], comp_input_queues[0]),
    (comp_threads[0], coproc_threads[0], coproc_input_queues[0]),
    (coproc_threads[0], comp_threads[0], coproc_completion_queues[0]),
]

# cpu_ratio is only used in weighted random LBs and ignored in other ones.
# Sangwook: It would be better to write 'cpu_ratio' only when it is needed, 
#            but it seems Python wrapper doesn't allow it..
LB_mode = str(os.environ.get('NSHADER_LOADBALANCER_MODE', 'CPUOnlyLB'))
LB_cpu_ratio = float(os.environ.get('NSHADER_LOADBALANCER_CPU_RATIO', 1.0))
load_balancer = nshader.LoadBalancer(mode=LB_mode, cpu_ratio=LB_cpu_ratio)
