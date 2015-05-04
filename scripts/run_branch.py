#! /usr/bin/env python3
import sys, time, os
import asyncio
import tempfile
from contextlib import ExitStack
from itertools import product
from statistics import mean
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
    lahti  = PktGenRunner('shader-lahti.anlab', 54321)
    pktgens = [marcel, lahti]

    iobatchsizes   = [32]
    compbatchsizes = [32]
    compppdepths   = [16]
    packetsize     = [64]#, 128, 256, 512, 1024, 1500]
    branch_configs = ["l2fwd-echo-branch-lv1.click"]#, "l2fwd-echo-branch-lv2.click", "l2fwd-echo-branch-lv3.click"]
    branch_ratio   = [50, 40, 30, 20, 10, 5, 1]

    print('Params: io-batch-size comp-batch-size comp-ppdepth pkt-size branch-lvl branch-ratio')
    print('Outputs: fwd-Mpps fwd-Gbps')

    for params in product(iobatchsizes, compbatchsizes, compppdepths,
                          packetsize, branch_configs, branch_ratio): #, coprocppdepths):

        sys.stdout.flush()
        env.envvars['NBA_IO_BATCH_SIZE'] = str(params[0])
        env.envvars['NBA_COMP_BATCH_SIZE'] = str(params[1])
        env.envvars['NBA_COMP_PPDEPTH'] = str(params[2])
        psize = str(params[3])
        branch_config = str(params[4])
        branch_ratio_local = params[5]

        # Configure what and how to measure things.
        thruput_fetcher = env.measure_thruput(begin_after=15.0, repeat=False)
        cpu_fetcher     = env.measure_cpu_usage(interval=2, begin_after=17.0, repeat=False)

        for pktgen in pktgens:
            pktgen.set_args("-i", "all", "-f", "0", "-v", "4", "-p", psize)

        high_branch = 100 - branch_ratio_local
        low_branch = branch_ratio_local

        # Generate Click config from a template
        click_path = os.path.normpath(os.path.join('configs', branch_config))
        temp_file  = tempfile.NamedTemporaryFile('w', prefix='nba.temp.click.', delete=False)
        with open(click_path, 'r') as infile, temp_file as outfile:
            data_in  = infile.read()
            data_out = data_in.format(str(high_branch), str(low_branch), 'echoback');
            print(data_out, file=outfile)

        # Run.
        with ExitStack() as stack:
            _ = [stack.enter_context(pktgen) for pktgen in pktgens]
            loop = asyncio.get_event_loop()
            loop.run_until_complete(env.execute_main('rss.py', temp_file.name, running_time=20.0))

        # Delete the generated config.
        os.unlink(temp_file.name)

        # Retrieve the results.
        print('{0:5} {1:5} {2:5} {3:5} {4:5} {5:5}'.format(*params), end=' ')

        # Fetch results of throughput measurement and compute average.
        average_thruput_mpps = mean(t.total_fwd_pps for t in thruput_fetcher)
        average_thruput_gbps = mean(t.total_fwd_bps for t in thruput_fetcher)
        print('{0:6.2f}'.format(average_thruput_mpps), end=' ')
        print('{0:6.2f}'.format(average_thruput_gbps), end=' ')
        print()

        # Check max result.
        sys.stdout.flush()
        time.sleep(3)
