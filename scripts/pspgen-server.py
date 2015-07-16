#! /usr/bin/env python3
import os, sys
import re
import asyncio, subprocess, signal
from functools import partial
import multiprocessing

MAX_DELAY_HISTORY = 24

class DelayReader():

    def __init__(self, parent):
        self._parent = parent
        self._rx_latency = re.compile(r"^CPU\s+\d:\s+[\d.]+\s+pps,\s+[\d.]+\s+Gbps\s+\([\d.]+\s+packets per chunk\)\s+(([\d.]+)\s+us\s+[\d.]+)?\s+xge\d+:\s+\d+\s+pps.*")
        self._delay_history = []

    @asyncio.coroutine
    def read(self):
        while True:
            line = yield from self._parent._proc.stdout.readline()
            if not line: break
            line = line.decode().strip()
            m = self._rx_latency.search(line)
            if m:
                delay_str = m.group(2)
                try:
                    delay = float(delay_str)
                    if len(self._delay_history) == MAX_DELAY_HISTORY:
                        self._delay_history.pop(0)
                    self._delay_history.append(delay)
                    avg_delay = sum(self._delay_history) / len(self._delay_history)
                    print('setting delay {0:.6f}'.format(avg_delay))
                    yield from self._parent.emit_delay(avg_delay)
                except (ValueError, TypeError):
                    pass
            else:
                print(line)


class PassthruReader():

    def __init__(self, parent):
        self._parent = parent

    @asyncio.coroutine
    def read(self):
        while True:
            line = yield from self._parent._proc.stdout.readline()
            if not line: break
            line = line.decode().strip()
            print(line)


class PspgenProxy:

    def __init__(self):
        self._running = False
        self._loop = None
        self._bin = './pspgen'
        self._writer = None

    @property
    def is_running(self):
        return self._running

    @asyncio.coroutine
    def emit_delay(self, value):
        self._writer.write('{0:.6f}\n'.format(value))
        yield from self._writer.drain()

    @asyncio.coroutine
    def start(self, loop, writer, args, read_latencies=False):
        if self._running:
            print('pspgen is already running.', file=sys.stderr)
            return
        self._running = True
        self._loop = loop
        self._writer = writer

        # Prepend the binary path and DPDK EAL arguments (for pspgen-dpdk).
        cpumask = 0
        for c in range(multiprocessing.cpu_count()):
            cpumask |= (1 << c)
        cmdargs = [self._bin] + ['-c', '{:x}'.format(cpumask), '-n', '4', '--'] + args

        # pspgen forks multiple children to generate packets in parallel.
        # The parent process ignores SIGINT by self._proc.send_signal() and its
        # children do not receive it!
        # Hence we make a new session group before exec() in the subprocess
        # children by setting preexec_fn to os.setsid and use os.killpg() to
        # ensure SIGINT is delivered to all pspgen subprocesses.
        print('Executing: {0}'.format(' '.join(cmdargs)))
        self._proc = yield from asyncio.create_subprocess_exec(*cmdargs, close_fds=True,
                                                               stdout=subprocess.PIPE,
                                                               stderr=subprocess.STDOUT,
                                                               start_new_session=True)
        if read_latencies:
            self._reader = DelayReader(self)
        else:
            self._reader = PassthruReader(self)
        asyncio.async(self._reader.read(), loop=self._loop)

    @asyncio.coroutine
    def terminate(self):
        if not self._running:
            print('pspgen is not running!', file=sys.stderr)
            return
        print('Terminating pspgen...')
        os.killpg(os.getsid(self._proc.pid), signal.SIGINT)
        exitcode = yield from self._proc.wait()
        self._running = False
        self._writer = None
        print('pspgen has terminated.')

pspgen_proxy = None
server_running = False

@asyncio.coroutine
def handle_request(loop, reader, writer):
    global pspgen_proxy, server_running
    # This program can only serve a single client at a time.
    assert not server_running
    server_running = True
    while True:
        line = yield from reader.readline()
        if not line: break
        line = line.strip().decode()
        args = line.split(':')
        if args[0] == 'start':
            read_latencies = (args[1] == 'latency')
            asyncio.async(pspgen_proxy.start(loop, writer, args[2:], read_latencies), loop=loop)
        elif args[0] == 'terminate':
            yield from pspgen_proxy.terminate()
        else:
            print('Unknown command: {0}'.format(args[0]), file=sys.stderr)
    server_running = False

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    pspgen_proxy = PspgenProxy()
    start_coro = asyncio.start_server(partial(handle_request, loop), '0.0.0.0', 54321, loop=loop)
    server = loop.run_until_complete(start_coro)
    print('Running...')
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print()
    print('Exit.')
    loop.close()
