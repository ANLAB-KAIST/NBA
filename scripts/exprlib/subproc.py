#! /usr/bin/env python3
import sys, os
import asyncio
import subprocess


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

_execute_memo = {}
def execute_memoized(cmd):
    if cmd in _execute_memo:
        return _execute_memo[cmd]
    else:
        ret = execute(cmd, shell=True, read=True)
        _execute_memo[cmd] = ret
        return ret
