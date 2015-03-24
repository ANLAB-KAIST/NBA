#! /usr/bin/env python3
import sys, os, pwd
import re
import asyncio
import argparse
import subprocess, signal
import multiprocessing
from pprint import pprint
from collections import defaultdict, namedtuple, OrderedDict
import ctypes, ctypes.util

_libnuma = ctypes.CDLL(ctypes.util.find_library('numa'))

RemoteServer = namedtuple('RemoteServer', [
    'hostname', 'username', 'keyfile', 'sshargs'
])
PortThruputRecord = namedtuple('PortThruputRecord', [
    'port', 'in_pps', 'in_bps', 'out_pps', 'out_bps', 'in_errs', 'out_errs',
])
ThruputRecord = namedtuple('ThruputRecord', [
    'timestamp',
    'avg_in_pps', 'avg_in_bps', 'avg_out_pps', 'avg_out_bps', 'avg_in_errs', 'avg_out_errs',
    'total_in_pps', 'total_out_pps', 'total_in_errs', 'total_out_errs',
    'per_port_stat', 'total_fwd_pps', 'total_fwd_bps',
])
CPURecord = namedtuple('CPURecord', [
    'timestamp',
    'core', 'usr', 'nice', 'sys', 'iowait', 'irq', 'soft', 'steal', 'guest', 'gnice', 'idle'
])

def execute(cmdargs, shell=False, iterable=False, async=False, read=False):
    '''
    Executes a shell command or external program using argument list.
    The output can be read after blocking or returned as iterable for further
    processing.
    This implementation is based on Snakemake's shell module.
    '''

    def _iter_stdout(proc, cmd):
        for line in proc.stdout:
            yield line[:-1]  # strip newline at the end
        retcode = proc.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, cmd)

    popen_args = {}
    if shell and 'SHELL' in os.environ:
        popen_args['executable'] = os.environ['SHELL']
    close_fds = (sys.platform != 'win32')
    stdout = subprocess.PIPE if (iterable or async or read) else sys.stdout
    proc = subprocess.Popen(cmdargs, shell=shell, stdout=stdout,
                            close_fds=close_fds, **popen_args)
    ret = None

    if iterable:
        return _iter_stdout(proc, cmdargs)
    if read:
        ret = proc.stdout.read()
    elif async:
        return proc
    retcode = proc.wait()
    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, cmdargs)
    return ret

@asyncio.coroutine
def execute_async_simple(cmdargs, timeout=None):
    proc = yield from asyncio.create_subprocess_exec(*cmdargs, stdout=sys.stdout)
    if timeout:
        try:
            retcode = yield from asyncio.wait_for(proc.wait(), timeout + 1)
        except asyncio.TimeoutError:
            print('Terminating the main process...', file=sys.stderr)
            proc.send_signal(signal.SIGINT)
            retcode = 0
    else:
        retcode = yield from proc.wait()
    return retcode

@asyncio.coroutine
def read_stdout_into_lines_coro(proc):
    lines = []
    while True:
        line = yield from proc.stdout.readline()
        if not line: break
        lines.append(line)
    return lines

def comma_sep_numbers(minval=0, maxval=sys.maxsize, type=int):
    def _comma_sep_argtype(string):
        try:
            pieces = list(map(lambda s: type(s.strip()), string.split(',')))
        except ValueError:
            raise argparse.ArgumentTypeError('{:r} contains non-numeric values.'.format(string))
        for p in pieces:
            if p < minval or p > maxval:
                raise argparse.ArgumentTypeError('{:r} contains a number out of range.'.format(string))
        return pieces
    return _comma_sep_argtype

def host_port_pair(default_port):
    def _host_port_pair_argtype(string):
        pieces = string.split(':')
        cnt = len(pieces)
        if cnt > 2:
            raise argparse.ArgumentTypeError('{:r} is not a valid host:port value.'.format(string))
        elif cnt == 2:
            try:
                host, port = pieces[0], int(pieces[1])
                assert port > 0 and port <= 65535
            except (ValueError, AssertionError):
                raise argparse.ArgumentTypeError('{:r} is not a valid port number.'.format(pieces[1]))
        else:
            host = pieces[0], default_port
        return host, port
    return _host_port_pair_argtype

_execute_memo = {}
def _exec_memoized(cmd):
    if cmd in _execute_memo:
        return _execute_memo[cmd]
    else:
        ret = execute(cmd, shell=True, read=True)
        _execute_memo[cmd] = ret
        return ret

class ExperimentEnv:

    rx_total_thruput = re.compile(r'^Total forwarded pkts: (?P<Mpps>\d+\.\d+) Mpps, (?P<Gbps>\d+\.\d+) Gbps in node (?P<node_id>\d+)$')
    rx_port_marker   = re.compile(r'^port\[(?P<node_id>\d+):(?P<port_id>\d+)\]')

    def __init__(self, verbose=False):
        self._verbose = verbose
        self._remote_servers = defaultdict(list)

        # Event loop states.
        self._main_cmdargs = None
        self._main_proc = None
        self._main_tasks = []
        self._last_cached_line = b''
        self._signal = None
        self._signalled = False
        self._finish_timer = None
        self._cpu_timer_fires = 0
        self._cpu_timer = None
        self._start_time = None
        self._loop = asyncio.get_event_loop()
        self._loop.add_signal_handler(signal.SIGINT, self._signal_coro)

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
        execute('chown -R {user}:{group} {0}'.format(
            path, user=ExperimentEnv.get_user(), group=ExperimentEnv.get_group(),
        ))

    @staticmethod
    def get_num_nodes():
        return int(_libnuma.numa_num_configured_nodes())

    @staticmethod
    def get_num_physical_cores():
        n = multiprocessing.cpu_count()
        if ExperimentEnv.is_hyperthreading():
            return n // 2
        return n

    @staticmethod
    def is_hyperthreading():
        siblings = _exec_memoized('cat /sys/devices/system/cpu/'
                                  'cpu0/topology/thread_siblings_list').decode('ascii').split(',')
        return len(siblings) > 1

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
        return int(execute('lspci | grep Ethernet | grep -c 82599', shell=True, read=True))

    @staticmethod
    def mangle_main_args(config_name, click_name, emulate_opts=None, extra_args=None):

        num_nodes = ExperimentEnv.get_num_nodes()
        num_ports = ExperimentEnv.get_num_ports()
        node_cpus = ExperimentEnv.get_core_topology()

        args = [
            '-c', 'f' * (multiprocessing.cpu_count() // 4),
            '-n', '4',
        ]

        if extra_args:
            args.extend(extra_args)
        args.append('--')
        args.append(config_name)
        args.append(click_name)
        if emulate_opts:
            emulate_args = {
                '--emulated-ipversion': 4,
                '--emulated-pktsize': 64,
                '--emulated-fixed-flows': 0,
            }
            emulate_args.update(emulate_opts)
            args.append('--emulate-io')
            for k, v in emulate_args.items():
                args.append('{0}={1}'.format(k, v))

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

    @asyncio.coroutine
    def execute_main(self, config_name, click_name, running_time=30.0, emulate_opts=None, extra_args=None):
        '''
        Executes the main program asynchronosuly.
        '''
        self.chdir_to_root()
        config_path = os.path.normpath(os.path.join('configs', config_name))
        click_path = os.path.normpath(os.path.join('configs', click_name))
        args = self.mangle_main_args(config_path, click_path, emulate_opts, extra_args)

        # Reset/initialize events.
        self._singalled = False
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
            print('Executing: bin/main', ' '.join(args))
        self._delayed_calls = []
        self._main_proc = yield from asyncio.create_subprocess_exec('bin/main', *args,
                                                                    stdout=subprocess.PIPE,
                                                                    env=self.get_merged_env())

        if self._main_proc.stderr is not None:
            self._main_tasks.append(self._main_read_stderr_coro(self._main_proc.stderr))
        assert(self._main_proc.stdout is not None)
        self._main_tasks.append(self._main_read_stdout_coro(self._main_proc.stdout))

        # Create timers.
        self._start_time = self._loop.time()
        self._delayed_calls.append(self._loop.call_later(running_time, self._main_finish_cb))
        self._cpu_timer_fires = 0
        if self._cpu_measure_time is not None:
            self._delayed_calls.append(self._loop.call_later(self._cpu_measure_time, self._cpu_timer_cb))

        # Run the stdout/stderr reader tasks.
        # (There may be other tasks in _main_tasks.)
        # Under normal conditions, these tasks will stop when _main_finish_cb() terminates the main process and thus reading stdout returns None.
        # Under abnormal conditions, asyncio.wait() call will time-out and we go into the reclaimation process below, which may use SIGKILL.
        if self._main_tasks:
            done, pending = yield from asyncio.wait(self._main_tasks, loop=self._loop, timeout=running_time + 1)

        # Reclaim the child process.
        try:
            exitcode = yield from asyncio.wait_for(self._main_proc.wait(), running_time+1)
        except asyncio.TimeoutError:
            # If the termination times out, kill it.
            # (Rarely some threads may hang up on termination...)
            # In this case, exitcode will be always -9.
            print('The main process hangs during termination. Killing it...', file=sys.stderr)
            self._main_proc.send_signal(signal.SIGKILL)
            exitcode = yield from self._main_proc.wait()
        self._main_proc = None
        self._main_tasks.clear()
        self._delayed_calls = None
        return exitcode

    def _print_timestamp(self, end=''):
        print('@{0:<11.6f} '.format(self._loop.time() - self._start_time), end=end)

    @asyncio.coroutine
    def _main_read_stderr_coro(self, stderr):
        while True:
            line = yield from stderr.readline()
            if not line: break
            # Currently we do nothing with stderr.

    @asyncio.coroutine
    def _main_read_stdout_coro(self, stdout):
        while True:
            line = yield from stdout.readline()
            if not line: break

            cur_ts = self._loop.time() - self._start_time
            if line.startswith(b'Total'):
                # Parse the line like "Total forwarded pkts: xx.xx Mpps, yy.yy Gbps in node x"
                line = line.decode('ascii')
                m = self.rx_total_thruput.search(line)
                if m is None: continue
                node_id = int(m.group('node_id'))
                self._total_mpps[node_id] = float(m.group('Mpps'))
                self._total_gbps[node_id] = float(m.group('Gbps'))
            elif line.startswith(b'port'):
                # Parse the lines like "port[x:x] x x x x .. | forwarded x.xx Mpps, y.yy Gbps"
                line = line.decode('ascii')
                m = self.rx_port_marker.search(line)
                if m is None: continue
                node_id = int(m.group('node_id'))
                port_id = int(m.group('port_id'))
                numbers = line.split(' | ')[0][m.end() + 1:]
                rx_pps, rx_bps, tx_pps, tx_bps, inv_pps, swdrop_pps, rxdrop_pps, txdrop_pps = \
                        map(lambda s: int(s.replace(',', '')), numbers.split())
                if port_id not in self._port_records:
                    self._port_records[port_id] = PortThruputRecord(port_id, rx_pps, rx_bps, tx_pps, tx_bps,
                                                              swdrop_pps + rxdrop_pps, txdrop_pps)
                else:
                    prev_record = self._port_records[port_id]
                    self._port_records[port_id] = PortThruputRecord(port_id,
                                                              prev_record.in_pps + rx_pps,
                                                              prev_record.in_bps + rx_bps,
                                                              prev_record.out_pps + tx_pps,
                                                              prev_record.out_bps + tx_bps,
                                                              prev_record.in_errs + swdrop_pps + rxdrop_pps,
                                                              prev_record.out_errs + txdrop_pps)

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

    def _main_finish_cb(self):
        if self._main_proc:
            self._main_proc.send_signal(signal.SIGINT)
        for handle in self._delayed_calls:
            handle.cancel()

    @asyncio.coroutine
    def _cpu_measure_coro(self):
        cur_ts = self._loop.time() - self._start_time
        if self._verbose:
            self._print_timestamp()
            print('Measuring CPU utilization...')
        proc = yield from asyncio.create_subprocess_shell(
            'mpstat -u -P ALL 1 2 | grep Average',
            stdout=subprocess.PIPE,
        )
        # Read all output at once.
        lines = yield from read_stdout_into_lines_coro(proc)
        exitcode = yield from proc.wait()
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

    @asyncio.coroutine
    def _signal_coro(self):
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

    def measure_thruput(self, ports='all', begin_after=15.0, repeat=False):
        '''
        Let the env instance to measure I/O throghput in pps during the loop.
        Since we can only depend on the statistics printed by the main program
        because DPDK does not have any system-common inspection interface like
        ethtool, the interval is strictly limited to 1 second (the stat update
        interval of the IO threads).
        '''
        # TODO: implement ports option?
        self._thruput_measure_time = begin_after
        self._thruput_measure_continuous = repeat
        return self._thruputs

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
        num_nodes = ExperimentEnv.get_num_nodes()
        num_coproc_cores = len(ExperimentEnv.get_coproc_cores())
        ht_div = 2 if ExperimentEnv.is_hyperthreading() else 1
        for n, cores in enumerate(node_cpus):
            core_set.extend(cores[:(len(cores) // ht_div - num_coproc_cores // num_nodes)])
        return core_set

    @staticmethod
    def get_comp_cores(layout_type):
        node_cpus = ExperimentEnv.get_core_topology()
        core_set = []
        num_nodes = ExperimentEnv.get_num_nodes()
        num_ports = ExperimentEnv.get_num_ports()
        num_coproc_cores = len(ExperimentEnv.get_coproc_cores())
        num_lcores_per_node = multiprocessing.cpu_count() // num_nodes
        if layout_type == 'same':
            for n, cores in enumerate(node_cpus):
                core_set.extend(cores[:len(cores) - num_coproc_cores])
        elif layout_type == 'sibling':
            assert ExperimentEnv.is_hyperthreading()
            for n, cores in enumerate(node_cpus):
                for c in cores[:len(cores) - num_coproc_cores]:
                    core_set.append(c + num_lcores_per_node)
        return core_set

    @staticmethod
    def get_coproc_cores():
        node_cpus = ExperimentEnv.get_core_topology()
        core_set = []
        num_nodes = ExperimentEnv.get_num_nodes()
        for n in range(num_nodes):
            core_set.append(node_cpus[n][-1])
        return core_set
