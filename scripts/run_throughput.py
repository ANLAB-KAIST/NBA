#! /usr/bin/env python3
import sys, os, time
import asyncio, signal
import argparse
from contextlib import ExitStack
from statistics import mean
from itertools import product
from exprlib import ExperimentEnv
from exprlib.subproc import execute_async_simple
from exprlib.arghelper import comma_sep_numbers, host_port_pair
from exprlib.records import AppThruputRecord, AppThruputReader
from exprlib.plotting.template import plot_thruput
from pspgen import PktGenRunner


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
    parser.add_argument('-p', '--pkt-sizes', type=comma_sep_numbers(64, 1500), metavar='NUM[,NUM...]', default=[64])
    parser.add_argument('--io-batch-sizes', type=comma_sep_numbers(1, 256), metavar='NUM[,NUM...]', default=[64])
    parser.add_argument('--comp-batch-sizes', type=comma_sep_numbers(1, 256), metavar='NUM[,NUM...]', default=[64])
    parser.add_argument('--coproc-ppdepths', type=comma_sep_numbers(1, 256), metavar='NUM[,NUM...]', default=[32])
    parser.add_argument('--pktgen', type=host_port_pair(54321), metavar='HOST:PORT[,HOST:PORT...]',
                        default=[('shader-marcel.anlab', 54321), ('shader-lahti.anlab', 54321)])
    parser.add_argument('--emulate-io', action='store_true', default=False, help='Use IO emulation mode.')
    parser.add_argument('-t', '--transparent', action='store_true', default=False, help='Pass-through the standard output instead of parsing it. No default timeout is applied.')
    parser.add_argument('--timeout', type=int, default=None, help='Set a forced timeout for transparent mode.')
    parser.add_argument('--combine-cpu-gpu', action='store_true', default=False, help='Run the same config for CPU-only and GPU-only to compare.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()

    env = ExperimentEnv(verbose=args.verbose)
    thruput_reader = AppThruputReader(begin_after=25.0)
    env.register_reader(thruput_reader)
    loop = asyncio.get_event_loop()
    all_thruput_records = []

    pktgens = []
    if not args.emulate_io:
        for host, port in args.pktgen:
            pktgens.append(PktGenRunner(host, port))

    if args.combine_cpu_gpu:
        base_conf_name, ext = os.path.splitext(os.path.basename(args.element_config_to_use))
        conf_names = [base_conf_name + '-cpuonly' + ext, base_conf_name + '-gpuonly' + ext]
    else:
        conf_names = [args.element_config_to_use]
    combinations = product(conf_names, args.io_batch_sizes, args.comp_batch_sizes, args.coproc_ppdepths, args.pkt_sizes)

    print('{0} {1}'.format(args.sys_config_to_use, args.element_config_to_use))
    if args.combine_cpu_gpu:
        print('conf  ', end='')
    print('io-batch-size comp-batch-size coproc-ppdepth pkt-size  ' \
          'Mpps Gbps  cpu-io-usr cpu-io-sys  cpu-coproc-usr cpu-coproc-sys')
    
    '''
    all_thruput_records.append(AppThruputRecord(64, 0, 'cpuonly', 27.0, 0))
    all_thruput_records.append(AppThruputRecord(64, 1, 'cpuonly', 28.2, 0))
    all_thruput_records.append(AppThruputRecord(1500, 0, 'cpuonly', 40.0, 0))
    all_thruput_records.append(AppThruputRecord(1500, 1, 'cpuonly', 39.9, 0))
    all_thruput_records.append(AppThruputRecord(64, 0, 'gpuonly', 13.0, 0))
    all_thruput_records.append(AppThruputRecord(64, 1, 'gpuonly', 15.2, 0))
    all_thruput_records.append(AppThruputRecord(1500, 0, 'gpuonly', 38.5, 0))
    all_thruput_records.append(AppThruputRecord(1500, 1, 'gpuonly', 39.7, 0))
    '''

    for conf_name, io_batch_size, comp_batch_size, coproc_ppdepth, pktsize in combinations:
        if args.combine_cpu_gpu:
            short_conf_name = 'cpuonly' if 'cpuonly' in conf_name else 'gpuonly'
            print(short_conf_name, end='  ')
        else:
            short_conf_name = os.path.splitext(os.path.basename(conf_name))[0]

        print('{0:<4d} {1:4d} {2:4d} {3:4d}'.format(io_batch_size, comp_batch_size, coproc_ppdepth, pktsize), end='  ')

        env.envvars['NBA_IO_BATCH_SIZE'] = str(io_batch_size)
        env.envvars['NBA_COMP_BATCH_SIZE'] = str(comp_batch_size)
        env.envvars['NBA_COPROC_PPDEPTH'] = str(coproc_ppdepth)

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

        #cpu_records     = env.measure_cpu_usage(interval=2, begin_after=26.0, repeat=True)

        # Clear data.
        env.reset_readers()
        thruput_reader.pktsize_hint = pktsize
        thruput_reader.conf_hint    = short_conf_name

        # Run.
        with ExitStack() as stack:
            _ = [stack.enter_context(pktgen) for pktgen in pktgens]
            if args.transparent:
                print('--- running in transparent mode ---')
                sys.stdout.flush()
                env.chdir_to_root()
                config_path = os.path.normpath(os.path.join('configs', args.sys_config_to_use))
                click_path = os.path.normpath(os.path.join('configs', conf_name))
                main_cmdargs = ['bin/main'] + env.mangle_main_args(config_path, click_path, emulate_opts=emulate_opts)
                retcode = loop.run_until_complete(execute_async_simple(main_cmdargs, timeout=args.timeout))
            else:
                retcode = loop.run_until_complete(env.execute_main(args.sys_config_to_use, conf_name,
                                                                   running_time=32.0, emulate_opts=emulate_opts))
            if retcode not in (0, -signal.SIGTERM, -signal.SIGKILL):
                print('The main program exited abnormaly, and we ignore the results! (exit code: {0})'.format(retcode))
                continue

        if args.transparent:
            continue

        # Fetch results of throughput measurement and compute average.
        num_nodes = env.get_num_nodes()
        thruput_records = thruput_reader.get_records()
        avg_thruput_mpps, avg_thruput_gbps = 0.0, 0.0
        for node_id in range(num_nodes):
            avg_thruput_mpps += mean(t.mpps for t in thruput_records if t.node_id == node_id)
            avg_thruput_gbps += mean(t.gbps for t in thruput_records if t.node_id == node_id)
        print('{0:6.2f}'.format(avg_thruput_mpps), end=' ')
        print('{0:6.2f}'.format(avg_thruput_gbps), end=' ')
        all_thruput_records.extend(thruput_records)

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
        print()

        sys.stdout.flush()
        time.sleep(3)

    loop.close()

    plot_thruput('app-thruput', all_thruput_records, args.element_config_to_use,
                 base_path='~/Dropbox/temp/plots/nba/',
                 combine_cpu_gpu=args.combine_cpu_gpu)

