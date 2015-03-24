#! /usr/bin/env python3
import nba, os

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
    'IO_BATCH_SIZE': int(os.environ.get('NBA_IO_BATCH_SIZE', 32)),
    'COMP_BATCH_SIZE': int(os.environ.get('NBA_COMP_BATCH_SIZE', 32)),
    'COMP_PPDEPTH': int(os.environ.get('NBA_COMP_PPDEPTH', 16)),
    'COPROC_PPDEPTH': int(os.environ.get('NBA_COPROC_PPDEPTH', 64)),
}
print("IO batch size: {0[IO_BATCH_SIZE]}, computation batch size: {0[COMP_BATCH_SIZE]}".format(system_params))
print("Computation pipeline depth: {0[COMP_PPDEPTH]}".format(system_params))
print("Coprocessor pipeline depth: {0[COPROC_PPDEPTH]}".format(system_params))
print("# logical cores: {0}, # physical cores {1} (hyperthreading {2})".format(
    nba.num_logical_cores, nba.num_physical_cores,
    "enabled" if nba.ht_enabled else "disabled"
))
_ht_diff = nba.num_physical_cores if nba.ht_enabled else 0

# The following objects are not "real" -- just namedtuple instances.
# They only store metdata w/o actual side-effects such as creation of threads.

no_cpu = int(os.environ.get('NBA_SINGLE_PORT_MULTI_CPU', 1))
no_node = int(os.environ.get('NBA_SINGLE_PORT_MULTI_CPU_NODE', 1))
no_port = int(os.environ.get('NBA_SINGLE_PORT_MULTI_CPU_PORT', 1))
print ("using " + str(no_cpu) + " cpus for " + str(no_port) + " port")
no_port_per_node = 4

io_threads = []
for node_id in range(no_node):
    for i in range(no_cpu):
        attached_rxq_gen = []
        for p in range(no_port):
            attached_rxq_gen.append((node_id*no_port_per_node + p, i))
        io_threads.append(nba.IOThread(core_id=node_cpus[node_id][i], attached_rxqs=attached_rxq_gen, mode='normal'))


comp_threads = []
for node_id in range(no_node):
    for i in range(no_cpu):
        comp_threads.append(nba.CompThread(core_id=node_cpus[node_id][i] + _ht_diff))

coproc_threads = []
for nid in range(no_node):
    # core_id, device_id
    coproc_threads.append(nba.CoprocThread(core_id=node_cpus[nid][7] + _ht_diff, device_id=nid))

comp_input_queues = []
for nid in range(no_node):
    for i in range(no_cpu):
        comp_input_queues.append(nba.Queue(node_id=nid, template='swrx'))

coproc_input_queues = []
for nid in range(no_node):
    coproc_input_queues.append(nba.Queue(node_id=nid, template='taskin'));

coproc_completion_queues = []
for node_id in range(no_node):
    for i in range(no_cpu):
        coproc_completion_queues.append(nba.Queue(node_id=node_id, template='taskout'))



queues = comp_input_queues + coproc_input_queues + coproc_completion_queues

thread_connections = []
for node_id in range(no_node):
    for i in range(no_cpu):
        thread_connections.append((io_threads[node_id*no_cpu + i], comp_threads[node_id*no_cpu + i], comp_input_queues[no_cpu*node_id + i]))

for node_id in range(no_node):
    for i in range(no_cpu):
        thread_connections.append((comp_threads[no_cpu * node_id + i], coproc_threads[node_id], coproc_input_queues[node_id]))
        
for node_id in range(no_node):
    for i in range(no_cpu):
        thread_connections.append((coproc_threads[node_id], comp_threads[no_cpu * node_id + i], coproc_completion_queues[no_cpu * node_id + i]))

# cpu_ratio is only used in weighted random LBs and ignored in other ones.
# Sangwook: It would be better to write 'cpu_ratio' only when it is needed, 
#            but it seems Python wrapper doesn't allow it..
LB_mode = str(os.environ.get('NBA_LOADBALANCER_MODE', 'CPUOnlyLB'))
LB_cpu_ratio = float(os.environ.get('NBA_LOADBALANCER_CPU_RATIO', 1.0))
load_balancer = nba.LoadBalancer(mode=LB_mode, cpu_ratio=LB_cpu_ratio)
