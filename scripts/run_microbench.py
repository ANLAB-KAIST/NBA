#! /usr/bin/env python3
import sys, time
from itertools import product
from exprlib import execute, ExperimentEnv
from pspgen import PktGenRunner

class ResultsAndConditions:
    def __init__(self):
        self._thruput = 0
        self._io_batch_size = 0
        self._comp_batch_size = 0
        self._comp_ppdepths = 0
        self._io_core_util_usr = 0
        self._io_core_util_sys = 0
        self._comp_core_util_usr = 0
        self._comp_core_util_sys = 0

    def is_thruput_bigger(self, thruput):
        if self._thruput < thruput:
            self._thruput = thruput
            return True
        return False

    def replace_contents(self, cond_io, cond_comp, cond_ppdepths,
                         result_io_usr, result_io_sys, result_comp_usr, result_comp_sys):
        self._io_batch_size = cond_io
        self._comp_batch_size = cond_comp
        self._comp_ppdepths = cond_ppdepths
        self._io_core_util_usr = result_io_usr
        self._io_core_util_sys = result_io_sys
        self._comp_core_util_usr = result_comp_usr
        self._comp_core_util_sys = result_comp_sys

    def print_result(self):
        print('Maximum throughput: {0:10}'.format(self._thruput))
        print('CPU utils: {0:6.2f} {1:6.2f} {2:6.2f} {3:6.2f}'.format(self._io_core_util_usr, self._io_core_util_sys,
                                                                      self._comp_core_util_usr, self._comp_core_util_sys))
        print('Conditions: {0:5} {1:5} {2:5}'.format(self._io_batch_size, self._comp_batch_size, self._comp_ppdepths))

if __name__ == '__main__':

    env = ExperimentEnv(verbose=False)
    marcel = PktGenRunner('shader-marcel.anlab', 54321)
    lahti = PktGenRunner('shader-lahti.anlab', 54321)
    #iobatchsizes = [8, 16, 32, 64, 128, 256]
    #compbatchsizes = [1, 8, 16, 32, 64, 128]

    iobatchsizes = [64]
    compbatchsizes = [64]
    coprocppdepths = [32]
    num_ports = [1, 2, 4]
    num_cores = [1, 2, 4, 8]
    num_nodes = [1, 2]
    packetsize = [60, 64, 128, 256, 512, 1024, 1500]

    #coprocppdepths = [16, 32, 64, 128]
    loadbalancers = ['CPUOnlyLB']

    print('NOTE: You should be running packet generators manually!')

    max_result = ResultsAndConditions()
    for lb_mode in loadbalancers:
        print('======== LoadBalancer mode: {0} ========='.format(lb_mode))
        print('Params: io-batch-size comp-batch-size comp-ppdepth')
        print('Outputs: forwarded-pps cpu-util(usr)-io cpu-util(sys)-io cpu-util(usr)-comp cpu-util(sys)-comp')
        sys.stdout.flush()
        env.envvars['NBA_LOADBALANCER_MODE'] = lb_mode 
        for params in product(iobatchsizes, compbatchsizes, num_ports, num_cores, num_nodes, packetsize): #, coprocppdepths):

            env.envvars['NBA_IO_BATCH_SIZE'] = str(params[0]) 
            env.envvars['NBA_COMP_BATCH_SIZE'] = str(params[1]) 
            env.envvars['NBA_SINGLE_PORT_MULTI_CPU_PORT'] = str(params[2])
            env.envvars['NBA_SINGLE_PORT_MULTI_CPU'] = str(params[3])
            env.envvars['NBA_SINGLE_PORT_MULTI_CPU_NODE'] = str(params[4])
            psize = str(params[5])

            # Configure what and how to measure things.
            thruput_fetcher = env.measure_thruput(begin_after=15.0, repeat=False)
            cpu_fetcher     = env.measure_cpu_usage(interval=1, begin_after=17.0, repeat=False)

            # Make progress of the generators to ensure configuration is applied.
            next(thruput_fetcher)
            next(cpu_fetcher)

            lahti.set_args("-i", "all", "-f", "0,", "-v", "4", "-p", psize)
            marcel.set_args("-i", "all", "-f", "0,", "-v", "4", "-p", psize)

            with lahti, marcel:
                # Run.
                env.execute_main('single-port-multi-cpu.py', sys.argv[1], running_time=20.0)

                # Retrieve the results.
                print('{0:5} {1:5} {2:5} {3:5} {4:5} {5:6}'.format(*params), end=' ')

                for thruput in thruput_fetcher:
                    print('totalpkt: {0:10,}'.format(thruput.total_fwd_pps), end=' ')

                    io_cores = env.get_io_cores(8)
                    comp_cores = env.get_comp_cores(8, 'same')
                    for cpu_usages in cpu_fetcher:
                        cpu_usage = tuple(filter(lambda o: str(o.core) == str(0), cpu_usages))
                        if cpu_usage:
                            print('{0:6.2f} {1:6.2f}'.format(cpu_usage[0].usr, cpu_usage[0].sys), end=' ')
                        cpu_usage = tuple(filter(lambda o: str(o.core) == str(16), cpu_usages))
                        if cpu_usage:
                            print('{0:6.2f} {1:6.2f}'.format(cpu_usage[0].usr, cpu_usage[0].sys))

                sys.stdout.flush()
                time.sleep(3)

        # Print max result.
        #max_result.print_result()
