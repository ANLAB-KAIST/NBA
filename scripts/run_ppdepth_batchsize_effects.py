#! /usr/bin/env python3
import sys
import asyncio
from itertools import product
from exprlib import execute, ExperimentEnv

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

    #iobatchsizes = [8, 16, 32, 64, 128, 256]
    #compbatchsizes = [1, 8, 16, 32, 64, 128]
    #compppdepths = [1, 8, 16, 32, 64, 128]
    iobatchsizes = [16, 32, 64, 128, 256]
    compbatchsizes = [8, 16, 32, 64, 128]
    compppdepths = [8, 16, 32, 64, 128]
    #coprocppdepths = [16, 32, 64, 128]
    loadbalancers = ['CPUOnlyLB', 'GPUOnlyLB']

    print('NOTE: You should be running packet generators manually!')

    max_result = ResultsAndConditions()
    for lb_mode in loadbalancers:
        print('======== LoadBalancer mode: {0} ========='.format(lb_mode))
        print('Params: io-batch-size comp-batch-size comp-ppdepth')
        print('Outputs: forwarded-pps cpu-util(usr)-io cpu-util(sys)-io cpu-util(usr)-comp cpu-util(sys)-comp')
        sys.stdout.flush()
        env.envvars['NSHADER_LOADBALANCER_MODE'] = lb_mode 
        for params in product(iobatchsizes, compbatchsizes, compppdepths): #, coprocppdepths):

            env.envvars['NSHADER_IO_BATCH_SIZE'] = str(params[0]) 
            env.envvars['NSHADER_COMP_BATCH_SIZE'] = str(params[1]) 
            env.envvars['NSHADER_COMP_PPDEPTH'] = str(params[2]) 

            # Configure what and how to measure things.
            thruput_fetcher = env.measure_thruput(begin_after=15.0, repeat=False)
            cpu_fetcher     = env.measure_cpu_usage(interval=1, begin_after=17.0, repeat=False)

            # Run.
            loop = asyncio.get_event_loop()
            loop.run_until_complete(env.execute_main('rss.py', sys.argv[1], running_time=20.0))

            # Retrieve the results.
            print('{0:5} {1:5} {2:5}'.format(params[0], params[1], params[2]), end=' ')

            for thruput in thruput_fetcher:
                print('{0:10,}'.format(thruput.avg_out_pps), end=' ')

                io_cores = env.get_io_cores(8)
                comp_cores = env.get_comp_cores(8, 'sibling')
                for cpu_usage in cpu_fetcher:
                    io_avg = (
                        sum(cpu_usage[core].usr for core in io_cores) / len(io_cores),
                        sum(cpu_usage[core].sys for core in io_cores) / len(io_cores),
                    )
                    comp_avg = (
                        sum(cpu_usage[core].usr for core in comp_cores) / len(comp_cores),
                        sum(cpu_usage[core].sys for core in comp_cores) / len(comp_cores),
                    )
                    print('{0:6.2f} {1:6.2f}'.format(io_avg[0], io_avg[1]), end=' ')
                    print('{0:6.2f} {1:6.2f}'.format(comp_avg[0], comp_avg[1]))

                # Check max result.
                if (max_result.is_thruput_bigger(thruput.avg_out_pps)):
                    max_result.replace_contents(params[0], params[1], params[2], io_avg[0], 
                                                io_avg[1], comp_avg[0], comp_avg[1])
            sys.stdout.flush()

        # Print max result.
        max_result.print_result()
