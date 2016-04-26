#! /usr/bin/env python3

import argparse
import asyncio
from datetime import datetime
import decimal
from itertools import product
import os
import signal
from statistics import mean
import sys
import time

from namedlist import namedlist
import numpy as np
import pandas as pd

from exprlib import ExperimentEnv
from exprlib.arghelper import comma_sep_numbers, comma_sep_str, host_port_pair
from exprlib.records import AppThruputRecord, AppThruputReader
from exprlib.plotting.utils import cdf_from_histogram
from exprlib.pktgen import PktGenController
from exprlib.latency import LatencyHistogramReader


ExperimentResult = namedlist('ExperimentResult', [
    ('thruput_records', None),
    ('latency_cdf', None),
])


def frange(start, end, interval):
    if (end - start) * interval < 0:
        raise ValueError
    decimal.getcontext().prec = 2
    _start      = decimal.Decimal(start)
    _end        = decimal.Decimal(end)
    _interval   = decimal.Decimal(interval)
    if interval == 0:
        return [start]
    else:
        n = int(round((end - start) / float(interval)))
        return [float(_start + _interval * i) for i in range(n + 1)]


async def do_experiment(loop, env, args, conds, thruput_reader):
    result = ExperimentResult()
    conf_name, pktsz, cpu_ratio = conds

    env.envvars['NBA_IO_BATCH_SIZE'] = '32'
    env.envvars['NBA_COMP_BATCH_SIZE'] = '64'
    env.envvars['NBA_COPROC_PPDEPTH'] = '32'
    env.envvars['NBA_LOADBALANCER_CPU_RATIO'] = cpu_ratio
    extra_nba_args = []

    offered_thruputs = {
        'ipv4': '10',
        'ipv6': '10',
        'ipsec': '3',
    }
    traffic_opts = {
        'ipv4': ['-v', '4', '-f', '0'],
        'ipv6': ['-v', '6', '-f', '0'],
        'ipsec': ['-v', '4', '-f', '1024', '-r', '0'],
    }
    if pktsz == 0:
        pktgen.args = ['-i', 'all', '--trace', 'traces/caida_anon_2016.pcap', '--repeat']
        if args.latency:
            if 'ipv6' in conf_name:
                pktgen.args += ['-g', offered_thruputs['ipv6'], '-l', '--latency-histogram']
            elif 'ipsec' in conf_name:
                extra_nba_args.append('--preserve-latency')
                pktgen.args += ['-g', offered_thruputs['ipsec'], '-l', '--latency-histogram']
            else:
                pktgen.args += ['-g', offered_thruputs['ipv4'], '-l', '--latency-histogram']
    else:
        if 'ipv6' in conf_name:
            # All random ipv6 pkts
            pktgen.args = ['-i', 'all'] + traffic_opts['ipv6'] + ['-p', str(pktsz)]
            if args.latency:
                pktgen.args += ['-g', offered_thruputs['ipv6'], '-l', '--latency-histogram']
        elif 'ipsec' in conf_name:
            # ipv4 pkts with fixed 1K flows
            pktgen.args = ['-i', 'all'] + traffic_opts['ipsec'] + ['-p', str(pktsz)]
            if args.latency:
                extra_nba_args.append('--preserve-latency')
                pktgen.args += ['-g', offered_thruputs['ipsec'], '-l', '--latency-histogram']
        else:
            # All random ipv4 pkts
            pktgen.args = ['-i', 'all'] + traffic_opts['ipv4'] + ['-p', str(pktsz)]
            if args.latency:
                pktgen.args += ['-g', offered_thruputs['ipv4'], '-l', '--latency-histogram']

    # Clear data.
    env.reset_readers()
    thruput_reader.pktsize_hint = pktsz
    thruput_reader.conf_hint    = conf_name

    # Run latency subscriber. (address = "tcp://generator-host:(54000 + cpu_idx)")
    # FIXME: get generator addr
    if args.latency:
        lhreader = LatencyHistogramReader(loop)
        hist_task = loop.create_task(lhreader.subscribe('shader-marcel.anlab', 0))

    async with pktgen:
        await asyncio.sleep(1)
        retcode = await env.execute_main(args.hw_config,
                                         args._conf_path_map[conf_name],
                                         extra_args=extra_nba_args,
                                         running_time=32.0)

    if args.latency:
        hist_task.cancel()
        await asyncio.sleep(0)

    # Fetch results of throughput measurement and compute average.
    # FIXME: generalize mean calculation
    thruput_records = thruput_reader.get_records()
    avg_thruput_records = []
    per_node_cnt = [0] * env.get_num_nodes()
    per_node_mpps_sum = [0.0] * env.get_num_nodes()
    per_node_gbps_sum = [0.0] * env.get_num_nodes()
    for r in thruput_records:
        per_node_cnt[r.node_id] += 1
        per_node_mpps_sum[r.node_id] += r.mpps
        per_node_gbps_sum[r.node_id] += r.gbps
    for n in range(env.get_num_nodes()):
        if per_node_cnt[n] > 0:
            avg_thruput_records.append((
                (conf_name, n, pktsz, cpu_ratio),
                (per_node_mpps_sum[n] / per_node_cnt[n],
                 per_node_gbps_sum[n] / per_node_cnt[n])))
    result.thruput_records = avg_thruput_records

    if args.latency:
        for r in reversed(lhreader.records):
            if len(r[2]) > 10:
                index, counts = zip(*(p.split() for p in r[2]))
                index  = np.array(list(map(int, index)))
                counts = np.array(list(map(int, counts)))
                result.latency_cdf = cdf_from_histogram(index, counts)
                break

    if retcode in (0, -signal.SIGTERM, -signal.SIGKILL):
        print(' .. case {0!r}: done.'.format(conds))
    else:
        print(' .. case {0!r}: exited abnormaly! (exit code: {1})'.format(conds, retcode))
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NOTE: 1. You must be running pspgen-servers '
                                                 'in the packet generator servers!\n'
                                                 '2. Packet size argument is only valid in emulation mode.',
                                     epilog='Example: sudo ./scriptname default.py ipv4-router.click 0 1 0.1',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('hw_config')
    parser.add_argument('element_configs', type=comma_sep_str(1), metavar='NAME[,NAME...]')
    parser.add_argument('-b', '--bin', type=str, metavar='PATH', default='bin/main')
    parser.add_argument('ratio_start', type=float)
    parser.add_argument('ratio_end', type=float)
    parser.add_argument('ratio_interval', type=float)
    parser.add_argument('--prefix', type=str, default=None,
                        help='Additional prefix directory name for recording.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--no-record', action='store_true', default=False,
                        help='Skip recording the results.')
    parser.add_argument('-l', '--latency', action='store_true', default=False,
                        help='Save the latency histogram.'
                             'The packet generation rate is fixed to'
                             '3 Gbps (for IPsec) or 10 Gbps (otherwise).')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--pkt-sizes', type=comma_sep_numbers(64, 1500),
                       metavar='NUM[,NUM...]', default=[64])
    group.add_argument('-t', '--trace', action='store_true', default=False)
    args = parser.parse_args()

    cpu_ratio_list = frange(args.ratio_start, args.ratio_end, args.ratio_interval)

    env = ExperimentEnv(verbose=args.verbose)
    env.main_bin = args.bin
    thruput_reader = AppThruputReader(begin_after=25.0)
    env.register_reader(thruput_reader)
    loop = asyncio.get_event_loop()

    pktgen = PktGenController()
    loop.run_until_complete(pktgen.init())

    conf_names = []
    args._conf_path_map = dict()
    for elem_config in args.element_configs:
        conf_name = os.path.splitext(os.path.basename(elem_config))[0]
        assert 'lbratio' in conf_name  # check config compatibility
        conf_path = elem_config
        conf_names.append(conf_name)
        args._conf_path_map[conf_name] = conf_path
    combinations = tuple(product(
        conf_names,
        tuple(range(env.get_num_nodes())),
        (0,) if args.trace else args.pkt_sizes,
        cpu_ratio_list
    ))
    combinations_without_node_id = tuple(product(
        conf_names,
        (0,) if args.trace else args.pkt_sizes,
        cpu_ratio_list
    ))
    mi = pd.MultiIndex.from_tuples(combinations, names=[
        'conf',
        'node_id',
        'pktsz',
        'cpu_ratio',
    ])
    all_tput_recs = pd.DataFrame(index=mi, columns=[
        'mpps',
        'gbps',
    ])
    all_latency_cdfs = dict()

    for conds in combinations_without_node_id:
        result = loop.run_until_complete(do_experiment(loop, env, args, conds, thruput_reader))
        for rec in result.thruput_records:
            all_tput_recs.ix[rec[0]] = rec[1]
        if result.latency_cdf is not None:
            all_latency_cdfs[conds] = result.latency_cdf
        sys.stdout.flush()
        time.sleep(3)
    loop.close()

    # Sum over node_id while preserving other indexes
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.float_format', lambda f: '{:.2f}'.format(f))
    system_tput = all_tput_recs.sum(level=['conf', 'pktsz', 'cpu_ratio'])
    print('Throughput per NUMA node')
    print('========================')
    print(all_tput_recs)
    print('Throughput per system')
    print('=====================')
    print(system_tput)
    print()
    if not args.no_record:
        now = datetime.now()
        bin_name = os.path.basename(args.bin)
        dir_name = 'loadbalance-ratio.{:%Y-%m-%d.%H%M%S}.{}'.format(now, bin_name)
        if args.prefix:
            dir_prefix = '{}.{:%Y-%m-%d}'.format(args.prefix, now)
            dir_name = os.path.join(dir_prefix, dir_name)
        base_path = os.path.join(os.path.expanduser('~/Dropbox/temp/plots/nba'), dir_name)
        os.makedirs(base_path, exist_ok=True)
        with open(os.path.join(base_path, 'version.txt'), 'w') as fout:
            print(env.get_current_commit(short=False), file=fout)
        all_tput_recs.to_csv(os.path.join(base_path, 'thruput.pernode.csv'), float_format='%.2f')
        system_tput.to_csv(os.path.join(base_path, 'thruput.csv'), float_format='%.2f')
        for conds, cdf in all_latency_cdfs.items():
            conds_str = '{5}'.format(*conds)  # only pktsz
            cdf.to_csv(os.path.join(base_path, 'latency.' + conds_str + '.csv'), float_format='%.6f')
        env.fix_ownership(base_path)
    print('all done.')
    sys.exit(0)
