#! /usr/bin/env python3
import sys, os, time
import asyncio
import argparse
import re
from contextlib import ExitStack
from collections import namedtuple
from datetime import datetime
from functools import partial
from itertools import product
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from exprlib import execute, execute_async_simple, comma_sep_numbers, host_port_pair, ExperimentEnv
from pspgen import PktGenRunner

PPCRecord = namedtuple('PPCRecord', 'ppc_cpu ppc_gpu ppc_est cpu_ratio thruput')

@asyncio.coroutine
def read_stdout_coro(records, stdout):
    rx_num = re.compile(r'\d+')
    rx_thr_gbps = re.compile(r'(\d+\.\d+) Gbps')
    rx_thr_node = re.compile(r'node (\d+)$')
    last_node_thruputs = [0, 0]
    while True:
        try:
            line = yield from stdout.readline()
        except asyncio.CancelledError:
            break
        if line is None: break
        line = line.decode('utf8')
        if line.startswith('[PPC:') and 'converge' not in line:
            pieces = tuple(s.replace(',', '') for s in line.split())
            node_id = int(rx_num.search(pieces[0]).group(0))
            thruput = last_node_thruputs[node_id]
            records[node_id].append(PPCRecord(int(pieces[2]), int(pieces[4]), int(pieces[6]), float(pieces[8]), thruput))
        if line.startswith('Total forwarded pkts'):
            node_id = int(rx_thr_node.search(line).group(1))
            gbps    = float(rx_thr_gbps.search(line).group(1))
            last_node_thruputs[node_id] = gbps

# Sample lines:
#   [PPC:1] CPU       30 GPU       47 PPC       30 CPU-Ratio 1.000
#   Total forwarded pkts: 39.53 Mpps, 27.83 Gbps in node 1

def draw_plot(records, confname, pktsize, base_path='~/Dropbox/temp/plots/nba/'):
    category = 'ppc-adaptive'
    now = datetime.now()
    timestamp_date = now.strftime('%Y-%m-%d')
    timestamp_full = now.strftime('%Y-%m-%d.%H%M%S')
    confname = Path(confname).stem
    path = Path(os.path.expanduser(base_path)) / '{0}.{1}'.format(category, timestamp_date) \
                                               / '{0}.{1}.pktsize-{2}.pdf'.format(timestamp_full, confname, pktsize)

    #plt.rc('font', family='Times New Roman')
    rec_count = min(len(recs) for recs in records)
    print('# records: {0}'.format(rec_count))

    fig, ax = plt.subplots()
    ax.set_xlim(0, rec_count)
    ax.set_xlabel('Ticks', fontweight='bold')
    ax.set_ylabel('Throughput (Gbps)', fontweight='bold')
    ax2 = ax.twinx()
    ax2.set_ylabel('Offloading Ratio', fontweight='bold')
    ax2.set_ylim(0, 1)

    line_colors = ['b', 'g']
    bar_colors = ['r', 'y']

    cpu_ratio = []
    thruput = []
    x_ind = np.arange(rec_count)

    for node_id in range(2):
        cpu_ratio.append(tuple(r.cpu_ratio for r in records[node_id][:rec_count]))
        thruput.append(tuple(r.thruput for r in records[node_id][:rec_count]))
        gpu_ratio = 1 - np.array(cpu_ratio[node_id][:rec_count])
        ax.bar(x_ind, thruput[node_id], bottom=thruput[node_id - 1] if node_id > 0 else None, color=bar_colors[node_id], edgecolor='none', width=1.05)
        ax2.plot(x_ind, gpu_ratio, color=line_colors[node_id])

    try:
        path.parent.mkdir(parents=True)
    except FileExistsError:
        pass
    print('Saving figure to {0}...'.format(path))
    plt.savefig(str(path), transparent=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(epilog='NOTE: 1. You must check pspgen-servers when you get ConnectionRefusedError!\n'
                                            '         They should be running in the pspgen directory\n'
                                            '         (e.g., ~/Packet-IO-Engine/samples/packet_generator$ ~/nba/scripts/pspgen-server.py)\n'
                                            '         at the packet generator servers.\n'
                                            '      2. Packet size argument is only valid in emulation mode.\n\n'
                                            'Example: sudo ./scriptname default.py ipv4-router.click\n ',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('sys_config_to_use')
    parser.add_argument('element_config_to_use')
    parser.add_argument('-p', '--pkt-sizes', type=comma_sep_numbers(64, 1500), metavar='NUM[,NUM...]', default=[64])
    parser.add_argument('--pktgen', type=host_port_pair(54321), metavar='HOST:PORT[,HOST:PORT...]',
                        default=[('shader-marcel.anlab', 54321), ('shader-lahti.anlab', 54321)])
    parser.add_argument('--emulate-io', action='store_true', default=False, help='Use IO emulation mode.')
    parser.add_argument('--timeout', type=int, default=None, help='Set a forced timeout for transparent mode.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()

    env = ExperimentEnv(verbose=args.verbose)
    pktgens = []
    if not args.emulate_io:
        for host, port in args.pktgen:
            pktgens.append(PktGenRunner(host, port))

    for pktsize in args.pkt_sizes:
        if 'ipv6' in args.element_config_to_use:
            # All random ipv6 pkts
            if args.emulate_io:
                emulate_opts = {'--emulated-pktsize': pktsize, '--emulated-ipversion': 6}
            else:
                emulate_opts = None
                for pktgen in pktgens:
                    pktgen.set_args("-i", "all", "-f", "0", "-v", "6", "-p", str(pktsize))
        elif 'ipsec' in args.element_config_to_use:
            # ipv4 pkts with 1K flows
            if args.emulate_io:
                emulate_opts = {'--emulated-pktsize': pktsize, '--emulated-fixed-flows': 1024}
            else:
                emulate_opts = None
                for pktgen in pktgens:
                    pktgen.set_args("-i", "all", "-f", "1024", "-r", "0", "-v", "4", "-p", str(pktsize))
        else:
            # All random ipv4 pkts
            if args.emulate_io:
                emulate_opts = {'--emulated-pktsize': pktsize}
            else:
                emulate_opts = None
                for pktgen in pktgens:
                    pktgen.set_args("-i", "all", "-f", "0", "-v", "4", "-p", str(pktsize))

        # Clear data
        records = [
            [], []
        ]

        # Run.
        with ExitStack() as stack:
            _ = [stack.enter_context(pktgen) for pktgen in pktgens]
            loop = asyncio.get_event_loop()
            retcode = loop.run_until_complete(env.execute_main(args.sys_config_to_use, args.element_config_to_use,
                                                               running_time=32.0, emulate_opts=emulate_opts,
                                                               custom_stdout_coro=partial(read_stdout_coro, records)))
            if retcode != 0 and retcode != -9:
                print('The main program exited abnormaly, and we ignore the results! (exit code: {0})'.format(retcode))
        
        #records[0].append(PPCRecord(10, 12, 10, 0.5, 25.5))
        #records[0].append(PPCRecord(11, 13, 12, 0.6, 23.5))
        #records[0].append(PPCRecord(10, 10, 13, 0.5, 22.5))
        #records[0].append(PPCRecord(11, 9, 15, 0.4, 28.5))
        #records[0].append(PPCRecord(11, 12, 16, 0.3, 29.5))

        sys.stdout.flush()
        time.sleep(1)

        draw_plot(records, args.element_config_to_use, pktsize)
