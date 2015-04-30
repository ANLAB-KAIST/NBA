#! /usr/bin/env python3
import os, sys
import re
import subprocess, signal
import multiprocessing
import threading
import socketserver

MAX_DELAY_HISTORY = 24

class DelayReader(threading.Thread):

    def __init__(self, parent):
        super(DelayReader, self).__init__()
        self._parent = parent
        self._rx_latency = re.compile(r"^CPU\s+\d:\s+[0123456789.]+\s+pps,\s+[0123456789.]+\s+Gbps\s+\([0123456789.]+\s+packets per chunk\)\s+(([0123456789.]+)\s+us\s+[0123456789.]+)?\s+xge\d+:\s+\d+\s+pps.*")
        self._delay_history = []
    
    def run(self):
        while True:
            line = self._parent._proc.stdout.readline()
            if not line:
                break
            self._parent._proc.poll()
            if self._parent._proc.returncode:
                break

            line = line.strip()
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
                    self._parent.emit_delay(avg_delay)
                except (ValueError, TypeError):
                    pass
            else:
                print(line)
        print('pspgen has terminated.')

class PspgenProxy:
    def __init__(self):
        self._running = False
        self._bin = './pspgen'
        self._wfile = None

    @property
    def is_running(self):
        return self._running

    def emit_delay(self, value):
        self._wfile.write('{0:.6f}\n'.format(value))
        self._wfile.flush()

    def start(self, args, output):
        if self._running:
            print('pspgen is already running.', file=sys.stderr)
            return
        self._running = True
        self._wfile = output

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
        self._proc = subprocess.Popen(cmdargs, close_fds=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT,
                                      preexec_fn=os.setsid,
                                      universal_newlines=True)
        self._reader = DelayReader(self)
        self._reader.start()

    def terminate(self):
        if not self._running:
            print('pspgen is not running!', file=sys.stderr)
            return
        print('Terminating pspgen...')
        os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
        self._proc.communicate()
        self._reader.join()
        self._running = False
        self._wfile = None

class PspgenServerHandler(socketserver.StreamRequestHandler):

    def handle(self):
        # This is a persistent handler.
        # Since the server and handlers run in the main thread synchronously,
        # this program can only serve a single client at a time.
        while True:
            line = self.rfile.readline()
            if not line:
                break
            line = line.strip().decode()
            args = line.split(':')
            if args[0] == 'start':
                self.server._pspgen_proxy.start(args[1:], self.wfile)
            elif args[0] == 'terminate':
                self.server._pspgen_proxy.terminate()
            else:
                print('Unknown command: {0}'.format(args[0]), file=sys.stderr)

if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(('0.0.0.0', 54321), PspgenServerHandler)
    server._pspgen_proxy = PspgenProxy()
    server.serve_forever()
