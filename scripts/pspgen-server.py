#! /usr/bin/env python3
import argparse
import os, sys
import re
import asyncio, subprocess, signal
from functools import partial
import multiprocessing
import aiozmq
import simplejson as json
import zmq

MAX_DELAY_HISTORY = 24

class DelayReader():

    def __init__(self, parent):
        self._parent = parent
        self._rx_latency = re.compile(r"^CPU\s+\d:\s+[\d.]+\s+pps,\s+[\d.]+\s+Gbps\s+\([\d.]+\s+packets per chunk\)\s+(([\d.]+)\s+us\s+[\d.]+)?\s+xge\d+:\s+\d+\s+pps.*")
        self._delay_history = []

    async def read(self):
        while True:
            line = await self._parent._proc.stdout.readline()
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
                    await self._parent.emit_delay(avg_delay)
                except (ValueError, TypeError):
                    pass
            else:
                print(line)


class PassthruReader():

    def __init__(self, parent):
        self._parent = parent

    async def read(self):
        while True:
            line = await self._parent._proc.stdout.readline()
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

    async def emit_delay(self, value):
        self._writer.write(['{0:.6f}\n'.format(value)])

    async def start(self, loop, args, read_latencies=False):
        if self._running:
            print('pspgen is already running.', file=sys.stderr)
            return
        self._running = True
        self._loop = loop
        self._writer = None
        #self._writer = await aiozmq.create_zmq_stream(zmq.PUSH, loop=loop, bind='tcp://*:54322')

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
        self._proc = await asyncio.create_subprocess_exec(*cmdargs, close_fds=True,
                                                          stdout=subprocess.PIPE,
                                                          stderr=subprocess.STDOUT,
                                                          start_new_session=True)
        if read_latencies:
            self._reader = DelayReader(self)
        else:
            self._reader = PassthruReader(self)
        asyncio.ensure_future(self._reader.read(), loop=self._loop)

    async def terminate(self):
        if not self._running:
            print('pspgen is not running!', file=sys.stderr)
            return
        print('Terminating pspgen...')
        os.killpg(os.getsid(self._proc.pid), signal.SIGINT)
        exitcode = await self._proc.wait()
        self._running = False
        if self._writer:
            self._writer.close()
        print('pspgen has terminated.')


async def command_loop(loop, controller_addr):
    pspgen_proxy = PspgenProxy()
    sock = await aiozmq.create_zmq_stream(zmq.SUB, loop=loop, connect=controller_addr)
    sock.transport.setsockopt(zmq.SUBSCRIBE, b'')
    while True:
        try:
            recv_data = await sock.read()
        except asyncio.CancelledError:
            sock.close()
            break
        except aiozmq.stream.ZmqStreamClosed:
            break
        msg = json.loads(recv_data[0])
        if msg['action'] == 'START':
            await pspgen_proxy.start(loop, msg['args'], msg['read_latencies'])
        elif msg['action'] == 'STOP':
            await pspgen_proxy.terminate()
        elif msg['action'] == 'NOOP':
            pass
        else:
            print('Unknown command: {0}'.format(msg['action']), file=sys.stderr)
    if pspgen_proxy.is_running:
        await pspgen_proxy.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('controller', type=str)
    args = parser.parse_args()
    print('Running...')
    loop = asyncio.get_event_loop()
    try:
        t = loop.create_task(command_loop(loop, args.controller))
        loop.run_forever()
    except KeyboardInterrupt:
        t.cancel()
        loop.run_until_complete(asyncio.sleep(0))
        print()
    print('Exit.')
    loop.close()
