#! /usr/bin/env python3
import sys, time, os
import asyncio
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
        print('CPU utils: {0:6.2f} {1:6.2f} {2:6.2f} {3:6.2f}'.format(self._io_core_util_usr, self._io_core_util_sys, self._comp_core_util_usr, self._comp_core_util_sys))
        print('Conditions: {0:5} {1:5} {2:5}'.format(self._io_batch_size, self._comp_batch_size, self._comp_ppdepths))

if __name__ == '__main__':

    env = ExperimentEnv(verbose=False)
    marcel = PktGenRunner('shader-marcel.anlab', 54321)

    iobatchsizes = [32]
    compbatchsizes = [32]
    compppdepths = [16]
    no_port = [4]
    no_cpu = [8]
    no_node = [2]
    packetsize = [64, 1500]#, 128, 256, 512, 1024, 1500]
    speed = [2,4,8,16] #,32,40,80]

    coprocppdepths = [16, 32, 64, 128]
    loadbalancers = ['CPUOnlyLB', 'GPUOnlyLB']

    print('NOTE: You should be running packet generators manually!')

    max_result = ResultsAndConditions()
    for lb_mode in loadbalancers:
        print('======== LoadBalancer mode: {0} ========='.format(lb_mode))
        print('Params: io-batch-size comp-batch-size comp-ppdepth')
        print('Outputs: forwarded-pps cpu-util(usr)-io cpu-util(sys)-io cpu-util(usr)-comp cpu-util(sys)-comp')
        sys.stdout.flush()
        env.envvars['NSHADER_LOADBALANCER_MODE'] = lb_mode
        for params in product(iobatchsizes, compbatchsizes, compppdepths,
                              no_port, no_cpu, no_node,
                              packetsize, speed, coprocppdepths):

            env.envvars['NSHADER_IO_BATCH_SIZE'] = str(params[0])
            env.envvars['NSHADER_COMP_BATCH_SIZE'] = str(params[1])
            env.envvars['NSHADER_COMP_PPDEPTH'] = str(params[2])
            env.envvars['NSHADER_SINGLE_PORT_MULTI_CPU_PORT'] = str(params[3])
            env.envvars['NSHADER_SINGLE_PORT_MULTI_CPU'] = str(params[4])
            env.envvars['NSHADER_SINGLE_PORT_MULTI_CPU_NODE'] = str(params[5])
            psize = str(params[6])
            speed_local = int(params[7])
            env.envvars['NSHADER_COPROC_PPDEPTH'] = str(params[8])

            # Configure what and how to measure things.
            thruput_fetcher = env.measure_thruput(begin_after=15.0, repeat=False)
            cpu_fetcher     = env.measure_cpu_usage(interval=1, begin_after=17.0, repeat=False)

            ip_version = '6' if 'ipv6' in app_name else '4'
            marcel.set_args("-i", "all", "-f", "0", "-v", ip_version, "-p", psize, "-g", speed_local, "-l", "1")
            with marcel:

                # Run.
                loop = asyncio.get_event_loop()
                loop.run_until_complete(env.execute_main('rss.py', "ipv6-router.click", running_time=20.0))

                # Retrieve the results.
                print('{0:5} {1:5} {2:5} {3:5} {4:5} {5:5} pkt {6:6} {7:7}Gbps cppdepth {8:8}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]), end=' ')

                for thruput in thruput_fetcher:
                    print('delay: {0:10}'.format(marcel.get_delay()), end=' ')
                    print('totalpkt: {0:10}'.format(thruput.total_fwd_pps), end=' ')
                    print('totaldrop: {0:10}'.format(thruput.total_in_errs), end=' ')

                    io_cores = env.get_io_cores(8)
                    comp_cores = env.get_comp_cores(8, 'sibling')
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
