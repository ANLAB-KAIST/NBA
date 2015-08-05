#! /usr/bin/env python3
import sys, os
from datetime import datetime
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
from matplotlib.gridspec import GridSpec
from ..records import PPCRecord, AppThruputRecord
from .utils import find_times_font


def kilo_formatter(val, idx):
    q = int(val) // 1000
    return str(q) + 'K' if q > 0 else '0'

def plot_ppc(category, records, confname, pktsize, base_path):
    now = datetime.now()
    timestamp_date = now.strftime('%Y-%m-%d')
    timestamp_full = now.strftime('%Y-%m-%d.%H%M%S')
    confname = Path(confname).stem
    path = Path(os.path.expanduser(base_path)) / '{0}.{1}'.format(category, timestamp_date) \
                                               / '{0}.{1}.pktsize-{2}.pdf'.format(timestamp_full, confname, pktsize)

    num_nodes = 2
    rec_counts = []
    for node_id in range(num_nodes):
        rec_counts.append(len([r for r in records if r.node_id == node_id]))
    rec_count = min(rec_counts)
    times_bold = find_times_font(bold=True)
    times_bold.set_size(6)
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 5,
        'axes.labelsize': 5,
        'axes.titlesize': 6,
        'legend.fontsize': 5,
        'axes.linewidth': 0.64,
    })

    fig = plt.figure(figsize=(3.3, 1.8))
    gs = GridSpec(2, 1, height_ratios=(4, 1.5))
    ax_perf = plt.subplot(gs[0])
    ax_ratio = plt.subplot(gs[1])

    ax_perf.set_xlim(0, rec_count - 1)
    ax_perf.set_ylim(0, 80)
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

    ax_ppc.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(kilo_formatter))

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

    for node_id in range(num_nodes):
        ppc_cpu = tuple(r.ppc_cpu for r in records if r.node_id == node_id)[:rec_count]
        ppc_gpu = tuple(r.ppc_gpu for r in records if r.node_id == node_id)[:rec_count]
        ppc_est = tuple(r.ppc_est for r in records if r.node_id == node_id)[:rec_count]
        thruput.append(tuple(r.thruput for r in records if r.node_id == node_id)[:rec_count])
        cpu_ratio.append(tuple(r.cpu_ratio for r in records if r.node_id == node_id)[:rec_count])
        offl_ratio = 1 - np.array(cpu_ratio[node_id][:rec_count])
        h_thr = ax_perf.bar(x_ind, thruput[node_id], bottom=thruput[node_id - 1] if node_id > 0 else None,
                            color=thr_colors[node_id], edgecolor=thr_colors[node_id], width=1.0, align='center')
        #step_thr = np.array(thruput[0])
        #for thr in thruput[1:node_id+1]:
        #    step_thr += np.array(thr)
        #ax_perf.step(x_ind, step_thr, color='black', where='mid')
        h_pcpu, = ax_ppc.plot(x_ind, ppc_cpu, color=ppc_colors[node_id], lw=0.3, alpha=0.72)
        h_pgpu, = ax_ppc.plot(x_ind, ppc_gpu, color=ppc_colors[node_id], lw=0.3, alpha=0.72)
        h_pest, = ax_ppc.plot(x_ind, ppc_est, color=ppc_colors[node_id], lw=0.8, alpha=0.5)
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


def plot_thruput(category, records, confname, base_path, combine_cpu_gpu=False, separate_nodes=False):
    now = datetime.now()
    timestamp_date = now.strftime('%Y-%m-%d')
    timestamp_full = now.strftime('%Y-%m-%d.%H%M%S')
    confname = Path(confname).stem
    path = Path(os.path.expanduser(base_path)) / '{0}.{1}'.format(category, timestamp_date) \
                                               / '{0}.{1}.pdf'.format(timestamp_full, confname)

    x_labels = sorted(set(t.pktsize for t in records))
    rec_count = len(x_labels)
    x_ind = np.arange(rec_count)
    times_bold = find_times_font(bold=True)
    times_bold.set_size(6)
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 5,
        'axes.labelsize': 5,
        'axes.titlesize': 6,
        'legend.fontsize': 5,
        'axes.linewidth': 0.64,
    })

    fig = plt.figure(figsize=(3.3, 1.8))
    gs = GridSpec(2, 1, height_ratios=(4, 1), wspace=0, hspace=0)
    ax_thruput = plt.subplot(gs[0])
    ax_legend  = plt.subplot(gs[1])
    ax_legend.axis('off')

    ax_thruput.set_xlabel('Packet sizes (bytes)', fontproperties=times_bold, labelpad=2)
    ax_thruput.set_ylabel('Throughput (Gbps)', fontproperties=times_bold, labelpad=2)
    ax_thruput.set_ylim(0, 80)

    thr_colors = [hex2color('#595959'), hex2color('#d6d6d6')]

    legends = []

    if separate_nodes:
        raise NotImplementedError()
    else:
        num_nodes = 2  # TODO: read from env
        if combine_cpu_gpu:
            cpu_thruputs = []
            gpu_thruputs = []
            for pktsize in x_labels:
                avg_cpu_thruput_gbps = 0.0
                avg_gpu_thruput_gbps = 0.0
                for node_id in range(num_nodes):
                    avg_cpu_thruput_gbps += mean(t.gbps for t in records if t.node_id == node_id and t.conf_hint == 'cpuonly' and t.pktsize == pktsize)
                    avg_gpu_thruput_gbps += mean(t.gbps for t in records if t.node_id == node_id and t.conf_hint == 'gpuonly' and t.pktsize == pktsize)
                cpu_thruputs.append(avg_cpu_thruput_gbps)
                gpu_thruputs.append(avg_gpu_thruput_gbps)
            w = 0.45
            h_cthr = ax_thruput.bar(x_ind,     cpu_thruputs, width=w, color=thr_colors[0], edgecolor='black', lw=0.64)
            h_gthr = ax_thruput.bar(x_ind + w, gpu_thruputs, width=w, color=thr_colors[1], edgecolor='black', lw=0.64)
            legends.append((h_cthr, 'CPU-only'))
            legends.append((h_gthr, 'GPU-only'))
            ax_thruput.set_xticks(x_ind + w)
            ax_thruput.set_xticklabels(x_labels)
        else:
            thruputs = []
            for pktsize in x_labels:
                avg_thruput_gbps = 0.0
                for node_id in range(num_nodes):
                    avg_thruput_gbps += mean(t.gbps for t in records if t.node_id == node_id and t.pktsize == pktsize)
                thruputs.append(avg_thruput_gbps)
            h_thr = ax_thruput.bar(x_ind, thruputs, width=0.8, color=thr_colors[0], edgecolor='black', lw=0.64)
            legends.append((h_thr, confname))
            ax_thruput.set_xticks(x_ind)
            ax_thruput.set_xticklabels(x_labels)

    legend = ax_legend.legend([li[0] for li in legends], [li[1] for li in legends], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.5))
    legend.get_frame().set_linewidth(0.64)

    try:
        path.parent.mkdir(parents=True)
    except FileExistsError:
        pass
    print('Saving figure to {0}...'.format(path))
    plt.savefig(str(path), bbox_inches='tight', transparent=True, dpi=300)

