#! /usr/bin/env python3
import sys, os, time
import asyncio, signal
import argparse
from contextlib import ExitStack
from datetime import datetime
from statistics import mean
from itertools import product

import pandas as pd

from exprlib import ExperimentEnv
from exprlib.subproc import execute_async_simple
from exprlib.arghelper import comma_sep_numbers, host_port_pair
from exprlib.records import AppThruputRecord, AppThruputReader
from exprlib.plotting.template import plot_thruput
from exprlib.pktgen import PktGenController
from exprlib.latency import LatencyHistogramReader


async def do_experiment(loop, env, args, conds, thruput_reader, all_tput_recs):
    conf_name, io_batchsz, comp_batchsz, coproc_ppdepth, pktsz = conds

    env.envvars['NBA_IO_BATCH_SIZE'] = str(io_batchsz)
    env.envvars['NBA_COMP_BATCH_SIZE'] = str(comp_batchsz)
    env.envvars['NBA_COPROC_PPDEPTH'] = str(coproc_ppdepth)

    if 'ipv6' in args.element_config_to_use:
        # All random ipv6 pkts
        pktgen.args = ['-i', 'all', '-f', '0', '-v', '6', '-p', str(pktsz)]
        if args.latency:
            pktgen.args += ['-g', '10', '-l', '--latency-histogram']
    elif 'ipsec' in args.element_config_to_use:
        # ipv4 pkts with fixed 1K flows
        pktgen.args = ['-i', 'all', '-f', '1024', '-r', '0', '-v', '4', '-p', str(pktsz)]
        if args.latency:
            pktgen.args += ['-g', '3', '-l', '--latency-histogram']
    else:
        # All random ipv4 pkts
        pktgen.args = ['-i', 'all', '-f', '0', '-v', '4', '-p', str(pktsz)]
        if args.latency:
            pktgen.args += ['-g', '10', '-l', '--latency-histogram']

    #cpu_records     = env.measure_cpu_usage(interval=2, begin_after=26.0, repeat=True)

    # Clear data.
    env.reset_readers()
    thruput_reader.pktsize_hint = pktsz
    thruput_reader.conf_hint    = conf_name

    # Run latency subscriber. (address = "tcp://generator-host:(54000 + cpu_idx)")
    # FIXME: get generator addr
    if args.latency:
        lhreader = LatencyHistogramReader(loop)
        hist_task = loop.create_task(lhreader.subscribe('shader-marcel.anlab', 0))

    # Run.
    async with pktgen:
        if args.transparent:
            print('--- running in transparent mode ---')
            sys.stdout.flush()
            env.chdir_to_root()
            config_path = os.path.normpath(os.path.join('configs', args.sys_config_to_use))
            click_path = os.path.normpath(os.path.join('configs', conf_name + '.click'))
            main_cmdargs = ['bin/main'] + env.mangle_main_args(config_path, click_path)
            retcode = await execute_async_simple(main_cmdargs, timeout=args.timeout)
        else:
            retcode = await env.execute_main(args.sys_config_to_use, conf_name + '.click', running_time=32.0)

    if args.latency:
        hist_task.cancel()
        await asyncio.sleep(0)

    if args.transparent:
        return

    # Fetch results of throughput measurement and compute average.
    # FIXME: generalize mean calculation
    thruput_records = thruput_reader.get_records()
    per_node_cnt = [0] * env.get_num_nodes()
    per_node_mpps_sum = [0.0] * env.get_num_nodes()
    per_node_gbps_sum = [0.0] * env.get_num_nodes()
    for r in thruput_records:
        per_node_cnt[r.node_id] += 1
        per_node_mpps_sum[r.node_id] += r.mpps
        per_node_gbps_sum[r.node_id] += r.gbps
    for n in range(env.get_num_nodes()):
        if per_node_cnt[n] > 0:
            all_tput_recs.ix[(conf_name, io_batchsz, comp_batchsz, coproc_ppdepth, n, pktsz)] \
                    = (per_node_mpps_sum[n] / per_node_cnt[n],
                       per_node_gbps_sum[n] / per_node_cnt[n])

    # TODO: Store latency histograms
    if args.latency:
        for r in reversed(lhreader.records):
            if len(r[2]) > 10:
                print(r)
                break

    ## Fetch results of cpu util measurement and compute average.
    #io_usr_avg = io_sys_avg = coproc_usr_avg = coproc_sys_avg = 0
    #io_usr_avgs = []; io_sys_avgs = []; coproc_usr_avgs = []; coproc_sys_avgs = []
    #io_cores = env.get_io_cores()
    #coproc_cores = env.get_coproc_cores()
    #for cpu_usage in cpu_records:
    #    io_usr_avgs.append(mean(cpu_usage[core].usr for core in io_cores))
    #    io_sys_avgs.append(mean(cpu_usage[core].sys for core in io_cores))
    #    coproc_usr_avgs.append(mean(cpu_usage[core].usr for core in coproc_cores))
    #    coproc_sys_avgs.append(mean(cpu_usage[core].sys for core in coproc_cores))
    #io_usr_avg = mean(io_usr_avgs)
    #io_sys_avg = mean(io_sys_avgs)
    #coproc_usr_avg = mean(coproc_usr_avgs)
    #coproc_sys_avg = mean(coproc_sys_avgs)
    #print('{0:6.2f} {1:6.2f}'.format(io_usr_avg, io_sys_avg), end='  ')
    #print('{0:6.2f} {1:6.2f}'.format(coproc_usr_avg, coproc_sys_avg))
    if retcode in (0, -signal.SIGTERM, -signal.SIGKILL):
        print(' .. case {0!r}: done.'.format(conds))
    else:
        print(' .. case {0!r}: exited abnormaly! (exit code: {1})'.format(conds, retcode))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(epilog='NOTE: 1. You must check pspgen-servers when you get ConnectionRefusedError!\n'
                                            '         They should be running in the pspgen directory\n'
                                            '         (e.g., ~/Packet-IO-Engine/samples/packet_generator$ ~/nba/scripts/pspgen-server.py)\n'
                                            '         at the packet generator servers.\n'
                                            '      2. Packet size argument is only valid in emulation mode.\n\n'
                                            'Example: sudo ./scriptname default.py ipv4-router.click\n ',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # TODO: Restrict program option: packet size argument is only valid in emulation mode.
    parser.add_argument('sys_config_to_use')
    parser.add_argument('element_config_to_use')
    parser.add_argument('-b', '--bin', type=str, metavar='PATH', default='bin/main')
    parser.add_argument('-p', '--pkt-sizes', type=comma_sep_numbers(64, 1500), metavar='NUM[,NUM...]', default=[64])
    parser.add_argument('--io-batch-sizes', type=comma_sep_numbers(1, 256), metavar='NUM[,NUM...]', default=[32])
    parser.add_argument('--comp-batch-sizes', type=comma_sep_numbers(1, 256), metavar='NUM[,NUM...]', default=[64])
    parser.add_argument('--coproc-ppdepths', type=comma_sep_numbers(1, 256), metavar='NUM[,NUM...]', default=[32])
    parser.add_argument('-t', '--transparent', action='store_true', default=False, help='Pass-through the standard output instead of parsing it. No default timeout is applied.')
    parser.add_argument('--timeout', type=int, default=None, help='Set a forced timeout for transparent mode.')
    parser.add_argument('--combine-cpu-gpu', action='store_true', default=False, help='Run the same config for CPU-only and GPU-only to compare.')
    parser.add_argument('-l', '--latency', action='store_true', default=False, help='Save the latency histogram.'
                                                                                    'The packet generation rate is fixed to'
                                                                                    '3 Gbps (for IPsec) or 10 Gbps (otherwise).')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()

    env = ExperimentEnv(verbose=args.verbose)
    env.main_bin = args.bin
    thruput_reader = AppThruputReader(begin_after=25.0)
    env.register_reader(thruput_reader)
    loop = asyncio.get_event_loop()

    pktgen = PktGenController()
    loop.run_until_complete(pktgen.init())

    base_conf_name = os.path.splitext(os.path.basename(args.element_config_to_use))[0]
    if args.combine_cpu_gpu:
        conf_names = [base_conf_name + '-cpuonly', base_conf_name + '-gpuonly']
    else:
        conf_names = [base_conf_name]
    combinations = tuple(product(
        conf_names,
        args.io_batch_sizes,
        args.comp_batch_sizes,
        args.coproc_ppdepths,
        tuple(range(env.get_num_nodes())),
        args.pkt_sizes
    ))
    combinations_without_node_id = tuple(product(
        conf_names,
        args.io_batch_sizes,
        args.comp_batch_sizes,
        args.coproc_ppdepths,
        args.pkt_sizes
    ))
    mi = pd.MultiIndex.from_tuples(combinations, names=[
        'conf',
        'io_batchsz',
        'comp_batchsz',
        'coproc_ppdepth',
        'node_id',
        'pktsz',
    ])
    all_tput_recs = pd.DataFrame(index=mi, columns=[
        'mpps',
        'gbps',
    ])

    '''
    _test_records = [
        AppThruputRecord(64, 0, 'ipsec-encryption-cpuonly', 27.0, 0),
        AppThruputRecord(64, 1, 'ipsec-encryption-cpuonly', 28.2, 0),
        AppThruputRecord(1500, 0, 'ipsec-encryption-cpuonly', 40.0, 0),
        AppThruputRecord(1500, 1, 'ipsec-encryption-cpuonly', 39.9, 0),
        AppThruputRecord(64, 0, 'ipsec-encryption-gpuonly', 13.0, 0),
        AppThruputRecord(64, 1, 'ipsec-encryption-gpuonly', 15.2, 0),
        AppThruputRecord(1500, 0, 'ipsec-encryption-gpuonly', 38.5, 0),
        AppThruputRecord(1500, 1, 'ipsec-encryption-gpuonly', 39.7, 0),
    ]
    for r in _test_records:
        all_tput_recs.ix[(r.conf, 32, 64, 32, r.node_id, r.pktsz)] = (r.mpps, r.gbps)
    '''
    for conds in combinations_without_node_id:
        loop.run_until_complete(do_experiment(loop, env, args, conds, thruput_reader, all_tput_recs))
        sys.stdout.flush()
        time.sleep(3)
    loop.close()

    # Sum over node_id while preserving other indexes
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.float_format', lambda f: '{:.2f}'.format(f))
    system_tput = all_tput_recs.sum(level=['conf', 'io_batchsz', 'comp_batchsz',
                                           'coproc_ppdepth', 'pktsz'])
    now = datetime.now()
    dir_name = 'apptput.{:%Y-%m-%d.%H%M%S}'.format(now)
    base_path = os.path.join(os.path.expanduser('~/Dropbox/temp/plots/nba'), dir_name)
    os.makedirs(base_path, exist_ok=True)
    base_filename = os.path.join(base_path, base_conf_name)
    print('Throughput per NUMA node')
    print('========================')
    print(all_tput_recs)
    print('Throughput per system')
    print('=====================')
    print(system_tput)
    print()
    with open(os.path.join(base_path, 'version.txt'), 'w') as fout:
        print(env.get_current_commit(short=False), file=fout)
    all_tput_recs.to_csv(os.path.join(base_filename + '.pernode.csv'), float_format='%.2f')
    system_tput.to_csv(os.path.join(base_filename + '.csv'), float_format='%.2f')
    #plot_thruput('apptput', all_tput_recs, args.element_config_to_use,
    #             base_path='~/Dropbox/temp/plots/nba/',
    #             combine_cpu_gpu=args.combine_cpu_gpu)
    env.fix_ownership(base_path)
    print('all done.')
    sys.exit(0)

