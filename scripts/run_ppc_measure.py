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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
from matplotlib.font_manager import findSystemFonts, FontProperties
from matplotlib.gridspec import GridSpec
from exprlib import execute, execute_async_simple, comma_sep_numbers, host_port_pair, ExperimentEnv
from pspgen import PktGenRunner

PPCRecord = namedtuple('PPCRecord', 'ppc_cpu ppc_gpu ppc_est cpu_ratio thruput')

@asyncio.coroutine
def read_stdout_coro(env, records, stdout):
    rx_num = re.compile(r'\d+')
    rx_thr_gbps = re.compile(r'(\d+\.\d+) Gbps')
    rx_thr_node = re.compile(r'node (\d+)$')
    last_node_thruputs = [0, 0]
    while True:
        line = yield from stdout.readline()
        if not line: break
        line = line.decode('utf8')
        if line.startswith('[MEASURE:'):
            pieces = tuple(s.replace(',', '') for s in line.split())
            node_id = int(rx_num.search(pieces[0]).group(0))
            thruput = last_node_thruputs[node_id]
            records[node_id].append(PPCRecord(int(pieces[2]), int(pieces[4]), int(pieces[6]), float(pieces[8]), thruput))
        if line.startswith('Total forwarded pkts'):
            node_id = int(rx_thr_node.search(line).group(1))
            gbps    = float(rx_thr_gbps.search(line).group(1))
            last_node_thruputs[node_id] = gbps
        if line.startswith('END_OF_TEST'):
            env.break_main()
            break

# Sample lines:
#   [MEASURE:1] CPU       30 GPU       47 PPC       30 CPU-Ratio 1.000
#   Total forwarded pkts: 39.53 Mpps, 27.83 Gbps in node 1

def find_times(bold=False, italic=False):
    fonts = findSystemFonts()
    for fontpath in fonts:
        fprop = FontProperties(fname=fontpath)
        name = fprop.get_name()
        name_matched = 'Times New Roman' in name
        pname = os.path.splitext(os.path.basename(fontpath))[0]
        style_matched = ((not bold) or (bold and (pname.endswith('Bold') or (pname.lower() == pname and pname.endswith('bd'))))) and \
                        ((not italic) or (italic and (pname.endswith('Italic') or (pname.lower() == pname and pname.endswith('i')))))
        if name_matched and style_matched:
            return fprop
    return None

def draw_plot(records, confname, pktsize, base_path='~/Dropbox/temp/plots/nba/'):
    category = 'ppc-measure'
    now = datetime.now()
    timestamp_date = now.strftime('%Y-%m-%d')
    timestamp_full = now.strftime('%Y-%m-%d.%H%M%S')
    confname = Path(confname).stem
    path = Path(os.path.expanduser(base_path)) / '{0}.{1}'.format(category, timestamp_date) \
                                               / '{0}.{1}.pktsize-{2}.pdf'.format(timestamp_full, confname, pktsize)

    #plt.rc('font', family='Times New Roman')
    rec_count = min(len(recs) for recs in records)
    print('# records: {0}'.format(rec_count))
    times_bold = find_times(True)
    times_bold.set_size(6)
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 5,
        'axes.labelsize': 5,
        'axes.titlesize': 6,
        'legend.fontsize': 5,
        'axes.linewidth': 0.64,
    })

    fig = plt.figure(figsize=(3.3, 1.8), dpi=150)
    gs = GridSpec(2, 1, height_ratios=(4, 1.5))
    ax_perf = plt.subplot(gs[0]) # sharex?
    ax_ratio = plt.subplot(gs[1])

    ax_perf.set_xlim(0, rec_count - 1)
    ax_perf.set_ylabel('   Throughput (Gbps)', fontproperties=times_bold, labelpad=2)
    ax_ppc = ax_perf.twinx()
    ax_ppc.set_ylabel('PPC (cycles)', fontproperties=times_bold, labelpad=3)
    ax_ppc.set_xlim(0, rec_count - 1)
    ax_ratio.set_xlabel('Ticks', fontproperties=times_bold, labelpad=1)
    ax_ratio.set_xlim(0, rec_count - 1)
    ax_ratio.set_ylabel('Offloading Weight', fontproperties=times_bold, labelpad=2)
    ax_ratio.set_ylim(0, 1)
    for l in ax_perf.get_xticklabels(): l.set_visible(False)
    for l in ax_ppc.get_xticklabels(): l.set_visible(False)
    l = ax_perf.get_yticklabels()[0]
    l.set_visible(False)

    box = ax_perf.get_position()
    ax_perf.set_position([box.x0 - 0.03, box.y0 + 0.05, box.width*0.62, box.height])
    box = ax_ppc.get_position()
    ax_ppc.set_position([box.x0 - 0.03, box.y0 + 0.05, box.width*0.62, box.height])
    box2 = ax_ratio.get_position()
    ax_ratio.set_position([box2.x0 - 0.03, box2.y0 + 0.05, box2.width*0.62, box.y0 - box2.y0])

    ppc_colors = [hex2color('#ff375d'), hex2color('#bf0000')]
    ratio_colors = [hex2color('#376bff'), hex2color('#0033bf')]
    thr_colors = [hex2color('#d8d8d8'), hex2color('#7f7f7f')]

    cpu_ratio = []
    thruput = []
    legend_items_thr = []
    legend_items_pcpu = []
    legend_items_pgpu = []
    legend_items_pest = []
    legend_items_ratio = []

    x_ind = np.arange(rec_count)

    for node_id in range(len(records)):
        ppc_cpu = tuple(r.ppc_cpu for r in records[node_id][:rec_count])
        ppc_gpu = tuple(r.ppc_gpu for r in records[node_id][:rec_count])
        ppc_est = tuple(r.ppc_est for r in records[node_id][:rec_count])
        thruput.append(tuple(r.thruput for r in records[node_id][:rec_count]))
        cpu_ratio.append(tuple(r.cpu_ratio for r in records[node_id][:rec_count]))
        offl_ratio = 1 - np.array(cpu_ratio[node_id][:rec_count])
        h_thr = ax_perf.bar(x_ind, thruput[node_id], bottom=thruput[node_id - 1] if node_id > 0 else None,
                            color=thr_colors[node_id], edgecolor=thr_colors[node_id], width=1.0, align='center')
        #step_thr = np.array(thruput[0])
        #for thr in thruput[1:node_id+1]:
        #    step_thr += np.array(thr)
        #ax_perf.step(x_ind, step_thr, color='black', where='mid')
        h_pcpu, = ax_ppc.plot(x_ind, ppc_cpu, color=ppc_colors[node_id], lw=0.8)
        h_pgpu, = ax_ppc.plot(x_ind, ppc_gpu, color=ppc_colors[node_id], lw=0.8)
        h_pest, = ax_ppc.plot(x_ind, ppc_est, color=ppc_colors[node_id], lw=1.6)
        h_ratio, = ax_ratio.plot(x_ind, offl_ratio, color=ratio_colors[node_id], lw=0.8)
        h_pcpu.set_dashes([0.6, 0.6])
        h_pgpu.set_dashes([2.4, 0.6])
        legend_items_thr.append((h_thr, 'Throughput (Node {0})'.format(node_id)))
        legend_items_pcpu.append((h_pcpu, 'PPC of CPU (Node {0})'.format(node_id)))
        legend_items_pgpu.append((h_pgpu, 'PPC of GPU (Node {0})'.format(node_id)))
        legend_items_pest.append((h_pest, 'Estimated PPC (Node {0})'.format(node_id)))
        legend_items_ratio.append((h_ratio, 'Offloading Weight (Node {0})'.format(node_id)))

    legend_items = legend_items_thr + legend_items_pcpu + legend_items_pgpu + legend_items_pest + legend_items_ratio
    legend = fig.legend([li[0] for li in legend_items], [li[1] for li in legend_items], bbox_to_anchor=(0.99, 0.5), loc='right', ncol=1, borderaxespad=0)
    legend.get_frame().set_linewidth(0.64)

    try:
        path.parent.mkdir(parents=True)
    except FileExistsError:
        pass
    print('Saving figure to {0}...'.format(path))
    plt.savefig(str(path), transparent=True, dpi=300)


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
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()

    env = ExperimentEnv(verbose=args.verbose)
    loop = asyncio.get_event_loop()
    pktgens = []
    if not args.emulate_io:
        for host, port in args.pktgen:
            pktgens.append(PktGenRunner(host, port))

    assert 'alb-measure' in args.element_config_to_use

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
            retcode = loop.run_until_complete(env.execute_main(args.sys_config_to_use, args.element_config_to_use,
                                                               running_time=0, emulate_opts=emulate_opts,
                                                               custom_stdout_coro=partial(read_stdout_coro, env, records)))
            if retcode != 0 and retcode != -9:
                print('The main program exited abnormaly, and we ignore the results! (exit code: {0})'.format(retcode))

        #records[0].append(PPCRecord(10, 12, 10, 0.5, 25.5))
        #records[0].append(PPCRecord(11, 13, 12, 0.6, 23.5))
        #records[0].append(PPCRecord(10, 10, 13, 0.5, 22.5))
        #records[0].append(PPCRecord(11, 9, 15, 0.4, 28.5))
        #records[0].append(PPCRecord(11, 12, 16, 0.3, 29.5))
        #records[1].append(PPCRecord(13, 10, 9, 0.3, 25.5))
        #records[1].append(PPCRecord(14, 15, 15, 0.25, 24.8))
        #records[1].append(PPCRecord(15, 11, 13, 0.4, 24.2))
        #records[1].append(PPCRecord(12, 12, 12, 0.48, 28.8))
        #records[1].append(PPCRecord(13, 13, 13, 0.5, 29.2))

        sys.stdout.flush()
        time.sleep(1)

        draw_plot(records, args.element_config_to_use, pktsize)

    loop.close()
