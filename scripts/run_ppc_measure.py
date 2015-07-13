#! /usr/bin/env python3
import sys, os, time
import asyncio, signal
import argparse
from contextlib import ExitStack
from exprlib import ExperimentEnv
from exprlib.arghelper import comma_sep_numbers, host_port_pair
from exprlib.records import PPCRecord, PPCReader
from exprlib.plotting.template import plot_ppc
from pspgen import PktGenRunner


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
    ppc_reader = PPCReader(env, prefix='MEASURE')
    env.register_reader(ppc_reader)
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

        # Clear data.
        env.reset_readers()

        # Run.
        with ExitStack() as stack:
            _ = [stack.enter_context(pktgen) for pktgen in pktgens]
            retcode = loop.run_until_complete(env.execute_main(args.sys_config_to_use, args.element_config_to_use,
                                                               running_time=0, emulate_opts=emulate_opts))
            if retcode not in (0, -signal.SIGTERM, -signal.SIGKILL):
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

        records = ppc_reader.get_records()
        plot_ppc('ppc-measure', records, args.element_config_to_use, pktsize, base_path='~/Dropbox/temp/plots/nba/')

    loop.close()
