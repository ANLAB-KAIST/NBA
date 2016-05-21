#! /usr/bin/env python3

import sys, os, pwd
import re
import asyncio
import argparse
import subprocess, shlex, signal
import multiprocessing
from pprint import pprint
from collections import defaultdict, namedtuple, OrderedDict
from .subproc import execute, execute_memoized
from .records import BaseReader
import ctypes, ctypes.util

_libnuma = ctypes.CDLL(ctypes.util.find_library('numa'))

RemoteServer = namedtuple('RemoteServer', [
    'hostname', 'username', 'keyfile', 'sshargs'
])
CPURecord = namedtuple('CPURecord', [
    'timestamp',
    'core', 'usr', 'nice', 'sys', 'iowait', 'irq', 'soft', 'steal', 'guest', 'gnice', 'idle'
])


async def read_stdout_into_lines_coro(proc):
    lines = []
    while True:
        line = await proc.stdout.readline()
        if not line: break
        lines.append(line)
    return lines


class ExperimentEnv:

    rx_port_marker   = re.compile(r'^port\[(?P<node_id>\d+):(?P<port_id>\d+)\]')

    def __init__(self, verbose=False):
        self._verbose = verbose
        self._remote_servers = defaultdict(list)

        # Event loop states.
        self._bin = 'bin/main'
        self._main_cmdargs = None
        self._main_proc = None
        self._main_tasks = []
        self._last_cached_line = b''
        self._signal = None
        self._signalled = False
        self._finish_timer = None
        self._break = False
        self._cpu_timer_fires = 0
        self._cpu_timer = None
        self._start_time = None
        self._readers = []
        self._loop = asyncio.get_event_loop()
        #self._loop.add_signal_handler(signal.SIGINT, self._signal_coro)

        # Configurations.
        self._nba_env = {
            'NBA_IO_BATCH_SIZE': 64,
            'NBA_COMP_BATCH_SIZE': 64,
            'NBA_COPROC_PPDEPTH': 32,
        }
        self._running_time = 0

        self._thruput_measure_time = None
        self._thruput_measure_continuous = False
        self._thruput_measured = False
        self._port_records = OrderedDict()
        self._total_mpps = OrderedDict()
        self._total_gbps = OrderedDict()

        self._cpu_measure_time = None
        self._cpu_measure_interval = 1.2
        self._cpu_measure_continuous = False
        self._cpu_measured = False

        # Results
        self._cpu = []
        self._thruputs = []

    @property
    def main_bin(self):
        return self._bin

    @main_bin.setter
    def main_bin(self, value):
        self._bin = value

    @staticmethod
    def get_user():
        return os.environ.get('SUDO_USER', os.environ['USER'])

    @staticmethod
    def get_group():
        return pwd.getpwnam(ExperimentEnv.get_user()).pw_gid

    @staticmethod
    def fix_ownership(path):
        '''
        Reset ownership of the given path.
        All experiments are executed as root, but analysis and data management
        is usually done as plain user accounts.  It is strongly suggested to
        call this function after data recording is finished.
        '''
        execute(['chown', '-R',
                 '{user}:{group}'.format(user=ExperimentEnv.get_user(), group=ExperimentEnv.get_group()),
                 '{0}'.format(path)])

    @staticmethod
    def get_num_nodes():
        return int(_libnuma.numa_num_configured_nodes())

    @staticmethod
    def get_num_physical_cores():
        n = multiprocessing.cpu_count()
        return n // ExperimentEnv.get_hyperthreading_degree()

    @staticmethod
    def get_hyperthreading_degree():
        siblings = execute_memoized('cat /sys/devices/system/cpu/'
                                  'cpu0/topology/thread_siblings_list').split(',')
        return len(siblings)

    @staticmethod
    def get_current_commit(short=True):
        commit = execute(['git', 'rev-parse', 'HEAD'], read=True).strip()
        if short:
            return commit[:7]
        return commit

    def add_remote_server(self, category, hostname,
                          username=None, keyfile=None, sshargs=None):
        '''
        Add a remote server into a category.
        '''
        if not username:
            username = ExperimentEnv.get_user()
        if not keyfile:
            keyfile = '{0}/.ssh/id_rsa'.format(os.environ['HOME'])
        if not sshargs:
            sshargs = []
        if not any(True for arg in sshargs
                   if arg.startswith('StrictHostKeyChecking')):
            sshargs.extend(['-o', 'StrictHostKeyChecking=no'])
        self._remote_servers[category].append(
            RemoteServer(hostname, username, keyfile, sshargs)
        )

    def execute_remote(self, category, cmdargs, wait_all=False):
        '''
        Executes the given command in parallel on all servers in the specified
        category.  Thie method is non-blocking by default, but you can set
        wait_all=True to wait until all commands are finished.
        '''
        ssh_procs = []
        for server in self._remote_servers[category]:
            ssh_cmdargs = [
                'ssh',
                '{0}@{1}'.format(server.username, server.hostname),
                '-i', server.keyfile,
            ]
            ssh_cmdargs.extend(server.sshargs)
            ssh_cmdargs.extend(cmdargs)
            proc = execute(ssh_cmdargs, async=True)
            ssh_procs.append(proc)
        if wait_all:
            while True:
                time.sleep(0.5)
                all_finished = True
                for proc in ssh_procs:
                    proc.poll()
                    all_finished &= (proc.returncode is not None)
                if all_finished:
                    break

    @staticmethod
    def get_num_ports():
        pmd = os.environ.get('NBA_PMD', 'ixgbe')
        if pmd == 'ixgbe':
            return int(execute('lspci | grep Ethernet | grep -c 82599', shell=True, read=True))
        elif pmd == 'mlx4' or pmd == 'mlnx_uio':
            return int(execute('lspci | grep Ethernet | grep -c Mellanox', shell=True, read=True))
        elif pmd == 'null':
            raise NotImplementedError()  # TODO: implement
        else:
            raise RuntimeError('Not recognized PMD: {0}'.format(pmd))

    @staticmethod
    def get_port_pci_addrs():
        pmd = os.environ.get('NBA_PMD', 'ixgbe')
        if pmd == 'ixgbe':
            return execute('lspci -D | grep Ethernet | grep 82599 | cut -d \' \' -f 1', shell=True, read=True).splitlines()
        elif pmd == 'mlx4' or pmd == 'mlnx_uio':
            return execute('lspci -D | grep Ethernet | grep Mellanox | cut -d \' \' -f 1', shell=True, read=True).splitlines()
        elif pmd == 'null':
            raise NotImplementedError()  # TODO: implement
        else:
            raise RuntimeError('Not recognized PMD: {0}'.format(pmd))

    @staticmethod
    def get_nodes_with_nics():
        '''
        Return a list of NUMA nodes that have NICs.
        '''
        pci_addrs = ExperimentEnv.get_port_pci_addrs()
        if len(pci_addrs) == 0:
            raise RuntimeError('No ports detected! Please check NBA_PMD environment variable.')
        core_bits = 0
        nodes_with_nics = set()
        for pci_addr in pci_addrs:
            sys_pci_path = '/sys/bus/pci/devices/{0}/numa_node'.format(pci_addr)
            with open(sys_pci_path, 'r', encoding='ascii') as f:
                nodes_with_nics.add(int(f.read()))
        return sorted(nodes_with_nics)

    @staticmethod
    def get_cpu_mask_with_nics(num_max_cores_per_node, only_phys_cores=True):
        '''
        Return a CPU core index bitmask of all cores in the NUMA nodes that have NICs.
        '''
        core_bits = 0
        node_cpus = ExperimentEnv.get_core_topology()
        nodes_with_nics = ExperimentEnv.get_nodes_with_nics()
        ht_div = ExperimentEnv.get_hyperthreading_degree() if only_phys_cores else 1
        for node_id in nodes_with_nics:
            phys_cnt = len(node_cpus[node_id]) // ht_div
            max_cnt = min(num_max_cores_per_node, phys_cnt)
            for core_id in node_cpus[node_id][:max_cnt]:
                core_bits |= (1 << core_id)
            # Force add device-handling cores
            core_bits |= (1 << node_cpus[node_id][phys_cnt - 1])
        return core_bits

    @staticmethod
    def mangle_main_args(config_name, click_name,
                         num_max_cores_per_node,
                         emulate_opts=None,
                         extra_args=None, extra_dpdk_args=None):
        core_mask = ExperimentEnv.get_cpu_mask_with_nics(num_max_cores_per_node)
        args = [
            '-c', '{:x}'.format(core_mask),
            '-n', os.environ.get('NBA_MEM_CHANNELS', '4'),
        ]
        # TODO: translate emulate_opts to void_pmd options
        if extra_dpdk_args:
            args.extend(extra_dpdk_args)
        args.append('--')
        if extra_args:
            args.extend(extra_args)
        args.append(config_name)
        args.append(click_name)
        return args

    @staticmethod
    def chdir_to_root():
        '''
        Changes the current working directory to the root of cloned repository.
        (TODO: .git may not be available in the deployed/distributed tarballs in the future!)
        Returns the relative path difference to the original script directory.
        '''
        root_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        my_abs_path = root_path
        while True:
            if os.path.isdir(os.path.join(root_path, '.git')) or root_path == '/':
                break
            root_path = os.path.join(*os.path.split(root_path)[:-1])
        os.chdir(root_path)
        return os.path.relpath(root_path, my_abs_path)

    async def execute_main(self, config_name, click_name,
                     running_time=30.0,
                     num_max_cores_per_node=64,
                     emulate_opts=None,
                     extra_args=None, extra_dpdk_args=None,
                     custom_stdout_coro=None):
        '''
        Executes the main program asynchronosuly.
        '''
        self.chdir_to_root()
        config_path = os.path.normpath(os.path.join('configs', config_name))
        click_path = os.path.normpath(os.path.join('configs', click_name))
        args = self.mangle_main_args(config_path, click_path, num_max_cores_per_node,
                                     emulate_opts, extra_args, extra_dpdk_args)

        # Reset/initialize events.
        self._singalled = False
        self._break = False
        self._thruput_measured = False
        self._cpu_measured = False
        self._running_time = running_time

        # Clear accumulated statistics.
        # (These objects may be referenced by the user,
        #  so we should only clear() instead of assigning a new list.)
        self._cpu.clear()
        self._thruputs.clear()

        # Run the main program.
        if self._verbose:
            print('Executing:', self._bin, ' '.join(args))
        self._delayed_calls = []

        if os.environ.get('NBA_USE_KNAPP', '0') == '1':
            subprocess.run("ssh -f mic0 \"sh -c 'nohup ./knapp-mic 2>&1 >/dev/null </dev/null &'\" 2>&1 >/dev/null", shell=True, check=True)

        self._main_proc = await asyncio.create_subprocess_exec(
            self._bin,
            *args,
            loop=self._loop,
            stdout=subprocess.PIPE,
            stderr=None,
            stdin=None,
            start_new_session=True,
            env=self.get_merged_env()
        )

        assert self._main_proc.stdout is not None

        # Run the readers.
        if custom_stdout_coro:
            asyncio.ensure_future(custom_stdout_coro(self._main_proc.stdout), loop=self._loop)
        else:
            asyncio.ensure_future(self._main_read_stdout_coro(self._main_proc.stdout), loop=self._loop)

        # Create timers.
        self._start_time = self._loop.time()
        self._cpu_timer_fires = 0
        if self._cpu_measure_time is not None:
            self._delayed_calls.append(self._loop.call_later(self._cpu_measure_time, self._cpu_timer_cb))

        # Wait.
        if running_time > 0:
            await asyncio.sleep(running_time)
        else:
            while not self._break:
                await asyncio.sleep(0.5)
        await asyncio.sleep(0.2)

        # Reclaim the child.
        try:
            sid = os.getsid(self._main_proc.pid)
            os.killpg(sid, signal.SIGTERM)
        except ProcessLookupError:
            # when the program automatically terminates (e.g., alb_measure)
            # it might already have terminated.
            pass
        for handle in self._delayed_calls:
            handle.cancel()
        try:
            exitcode = await asyncio.wait_for(self._main_proc.wait(), 3)
        except asyncio.TimeoutError:
            # If the termination times out, kill it.
            # (GPU/ALB configurations often hang...)
            print('The main process hangs during termination. Killing it...', file=sys.stderr)
            os.killpg(os.getsid(self._main_proc.pid), signal.SIGKILL)
            exitcode = -signal.SIGKILL
            # We don't wait it...

        if os.environ.get('NBA_USE_KNAPP', '0') == '1':
            subprocess.run("ssh mic0 'killall -9 knapp-mic'", shell=True, check=True)
        self._main_proc = None
        self._main_tasks.clear()
        self._delayed_calls = None
        self._break = False
        return exitcode

    def _print_timestamp(self, end=''):
        print('@{0:<11.6f} '.format(self._loop.time() - self._start_time), end=end)

    def break_main(self):
        self._break = True

    async def _main_read_stderr_coro(self, stderr):
        while True:
            line = await stderr.readline()
            if not line: break
            # Currently we do nothing with stderr.

    async def _main_read_stdout_coro(self, stdout):
        while True:
            line = await stdout.readline()
            if not line: break
            line = line.decode('utf8')
            cur_ts = self._loop.time() - self._start_time
            for reader in self._readers:
                reader.parse_line(cur_ts, line)

    '''
            # If we collected "Total forwarded Mpps" for all nodes, generate the stat.
            if len(self._total_mpps) == ExperimentEnv.get_num_nodes():
                # condition: continuous or not? && after some time or not?
                if (self._thruput_measure_continuous \
                       or (self._thruput_measure_continuous is False and not self._thruput_measured)) \
                   and ((self._thruput_measure_time is None) \
                       or (self._thruput_measure_time is not None and cur_ts > self._thruput_measure_time)):
                    tr = ThruputRecord(
                        cur_ts,
                        sum(r.in_pps   for r in self._port_records.values()) // len(self._port_records),
                        sum(r.in_bps   for r in self._port_records.values()) // len(self._port_records),
                        sum(r.out_pps  for r in self._port_records.values()) // len(self._port_records),
                        sum(r.out_bps  for r in self._port_records.values()) // len(self._port_records),
                        sum(r.in_errs  for r in self._port_records.values()) // len(self._port_records),
                        sum(r.out_errs for r in self._port_records.values()) // len(self._port_records),
                        sum(r.in_pps   for r in self._port_records.values()),
                        sum(r.out_pps  for r in self._port_records.values()),
                        sum(r.in_errs  for r in self._port_records.values()),
                        sum(r.out_errs for r in self._port_records.values()),
                        self._port_records,
                        sum(self._total_mpps.values()),
                        sum(self._total_gbps.values()),
                    )
                    self._thruput_measured = True
                    self._thruputs.append(tr)
                    if self._verbose:
                        self._print_timestamp()
                        print("Average: RX {0:10,} pps, TX {1:10,} pps".format(tr.avg_in_pps, tr.avg_out_pps))
                self._total_mpps.clear()
                self._total_gbps.clear()
                self._port_records.clear()
    '''

    async def _cpu_measure_coro(self):
        cur_ts = self._loop.time() - self._start_time
        if self._verbose:
            self._print_timestamp()
            print('Measuring CPU utilization...')
        proc = await asyncio.create_subprocess_shell(
            'mpstat -u -P ALL 1 2 | grep Average',
            stdout=subprocess.PIPE,
        )
        # Read all output at once.
        lines = await read_stdout_into_lines_coro(proc)
        exitcode = await proc.wait()
        assert exitcode == 0
        # Generate the stat.
        cpu_records = OrderedDict()
        for idx, line in enumerate(lines):
            if idx == 0: continue  # skip the header line
            pieces = line.split()
            # core may be "all" or zero-based index.
            core = int(pieces[1]) if pieces[1].isdigit() else pieces[1]
            values = tuple(map(float, pieces[2:]))
            record = CPURecord(cur_ts, core, *values)
            cpu_records[core] = record
        self._cpu.append(cpu_records)

    def _cpu_timer_cb(self):
        # Schedule the measurement coroutine.
        asyncio.async(self._cpu_measure_coro(), loop=self._loop)
        self._cpu_timer_fires += 1
        next_schedule_time = self._loop.time() - self._start_time + self._cpu_measure_interval
        if self._cpu_measure_continuous and next_schedule_time < self._running_time - 1.1:
            # This accumulates subsequent repeated calls, but we don't
            # remove/replace expired handles since there will be not many ones
            # and managing them correctly incurs annoyance of tracking.
            self._delayed_calls.append(self._loop.call_later(self._cpu_measure_interval, self._cpu_timer_cb))

    async def _signal_coro(self):
        if self._signalled: return
        self._signalled = True
        if self._main_proc:
            self._main_proc.send_signal(signal.SIGINT)
            if self._verbose:
                print()
                self._print_timestamp()
                print('Forcing termination of the main program and the event loop.')

    @property
    def envvars(self):
        return self._nba_env

    def get_merged_env(self):
        # Copy the inherited env-vars and add our specific ones.
        os_env = {k: v for k, v in os.environ.items()}
        os_env.update({k: str(v) for k, v in self._nba_env.items()})
        return os_env

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def register_reader(self, reader):
        assert isinstance(reader, BaseReader)
        self._readers.append(reader)

    def reset_readers(self):
        for reader in self._readers:
            reader.reset()

    def measure_cpu_usage(self, cores='all', interval=1, begin_after=15.0, repeat=False):
        '''
        Let the env instance to measure CPU usage in percent during the loop.
        '''
        # TODO: implement cores option?
        assert interval > 1.01
        self._cpu_measure_interval = float(interval)
        self._cpu_measure_time = begin_after
        self._cpu_measure_continuous = repeat
        return self._cpu

    @staticmethod
    def get_core_topology():
        node_cpus = []
        for n in range(ExperimentEnv.get_num_nodes()):
            node_cpus.append([])
        for c in range(multiprocessing.cpu_count()):
            node_id = int(_libnuma.numa_node_of_cpu(ctypes.c_int(c)))
            node_cpus[node_id].append(c)
        return node_cpus

    @staticmethod
    def get_io_cores():
        core_set = []
        node_cpus = ExperimentEnv.get_core_topology()
        for node_id in ExperimentEnv.get_nodes_with_nics():
            coproc_cores = ExperimentEnv.get_coproc_cores_in_node(node_id)
            phys_cnt = len(node_cpus[node_id]) // ht_div
            if len(coproc_cores) > 0:
                core_set.extend(node_cpus[node_id][:phys_cnt - len(coproc_cores)])
            else:
                core_set.extend(node_cpus[node_id][:phys_cnt])
        return core_set

    @staticmethod
    def get_comp_cores(layout_type):
        core_set = []
        node_cpus = ExperimentEnv.get_core_topology()
        ht_div = ExperimentEnv.get_hyperthreading_degree()
        if layout_type == 'same':
            for node_id in ExperimentEnv.get_nodes_with_nics():
                coproc_cores = ExperimentEnv.get_coproc_cores_in_node(node_id)
                phys_cnt = len(node_cpus[node_id]) // ht_div
                if len(coproc_cores) > 0:
                    core_set.extend(node_cpus[node_id][:phys_cnt - len(coproc_cores)])
                else:
                    core_set.extend(node_cpus[node_id][:phys_cnt])
        elif layout_type == 'sibling':
            assert ht_div > 1
            for node_id in ExperimentEnv.get_nodes_with_nics():
                coproc_cores = ExperimentEnv.get_coproc_cores_in_node(node_id)
                phys_cnt = len(node_cpus[node_id]) // ht_div
                if len(coproc_cores) > 0:
                    core_set.extend(c + phys_cnt for c in node_cpus[node_id][:phys_cnt - len(coproc_cores)])
                else:
                    core_set.extend(c + phys_cnt for c in node_cpus[node_id][:phys_cnt])
        return core_set

    @staticmethod
    def get_coproc_cores():
        # FIXME: get actual coprocessor count instead of assuming one per node.
        core_set = []
        node_cpus = ExperimentEnv.get_core_topology()
        nodes_with_nics = ExperimentEnv.get_nodes_with_nics()
        for node_id in nodes_with_nics:
            core_set.append(node_cpus[node_id][-1])
        return core_set
