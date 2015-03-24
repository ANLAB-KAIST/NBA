#! /usr/bin/env python3
from itertools import product
from exprlib import execute, ExperimentEnv
import subprocess, sys, os, time
import argparse

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('hardware_config',
                           help='The hardware configuration filename (python scripts).')
    argparser.add_argument('pipeline_config',
                           help='The pipeline configuration filename (.click files).')
    argparser.add_argument('-p', '--packet-size', type=int, default=60,
                           help='The packet size to generate in packet generators. (default: 60 bytes)')
    argparser.add_argument('-v', '--ip-version', type=int, choices=(4, 6), default=4,
                           help='The IP version to generate in packet generators. (default: 4)')
    argparser.add_argument('-g', '--gen-server', nargs='*', default=['shader-marcel.anlab', 'shader-lahti.anlab'],
                           help='The generator hostnames to use. (default: shader-marcel.anlab and shader-lahti.anlab)')
    args = argparser.parse_args()

    if os.geteuid() != 0:
        print('You must be root!', file=sys.stderr)
        sys.exit(1)

    env = ExperimentEnv(verbose=False)
    for pktgen_hostname in args.gen_server:
        env.add_remote_server('pktgen', pktgen_hostname)
    my_hostname = execute(['uname', '-n'], read=True).decode('ascii').strip().split('-')[-1]

    env.execute_remote('pktgen', ['Packet-IO-Engine/samples/packet_generator/pspgen '
                                  '-i all -f 0 -v {0} -p {1} '
                                  '--neighbor-conf Packet-IO-Engine/samples/packet_generator/neighbors-in-{2}.conf'
                                  ' > /dev/null 2>&1'.format(
                                    args.ip_version, args.packet_size, my_hostname,
                                  )])

    cmdargs = env.mangle_main_args(args.hardware_config, args.pipeline_config)
    proc = subprocess.Popen(['bin/main'] + cmdargs)
    proc.communicate()

    time.sleep(3)
    env.execute_remote('pktgen', ['killall pspgen'])

