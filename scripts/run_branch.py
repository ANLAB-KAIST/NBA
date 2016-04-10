#! /usr/bin/env python3
import sys, os
import subprocess
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--branch-pred-type', choices=('on', 'off', 'always'), required=True)
    args = parser.parse_args()
    args.bin = 'bin-backup/main.branchpred.' + args.branch_pred_type

    #branch_configs = ["l2fwd-echo-branch-lv1.click"]#, "l2fwd-echo-branch-lv2.click", "l2fwd-echo-branch-lv3.click"]
    branch_config = 'l2fwd-echo-skewed-branch-lv3.click'
    #branch_ratios = [50, 40, 30, 20, 10, 5, 1]
    branch_ratios = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]

    conf_paths = []
    os.chdir('..')

    # Generate templated configs
    for ratio in branch_ratios:
        p = os.path.splitext(branch_config)
        gen_path = p[0] + '-ratio{:02d}'.format(ratio) + p[1]
        conf_paths.append(gen_path)

        high_branch = '{:.2f}'.format((100 - ratio) / 100)
        low_branch  = '{:.2f}'.format((ratio) / 100)
        with open(os.path.join('configs', branch_config), 'r') as infile, \
             open(os.path.join('configs', gen_path), 'w') as outfile:
            data_in  = infile.read()
            data_out = data_in.format(high_branch, low_branch, 'echoback');
            print(data_out, file=outfile, end='')

    os.chdir('scripts')
    # Run.
    subprocess.run(['dropbox.py', 'stop'])
    subprocess.run([
        './run_app_perf.py',
        '--prefix', 'branch-pred.' + args.branch_pred_type,
        '-b', args.bin,
        '-p', '64',
        'default.py',
        ','.join(conf_paths),
    ])
    subprocess.run(['dropbox.py', 'start'])

    # Delete templated configs.
    os.chdir('..')
    for path in conf_paths:
        os.unlink(os.path.join('configs', path))
