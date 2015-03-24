#! /usr/bin/env python3
'''
Created on 2014. 1. 29.

@author: leeopop
'''

import sys, time
import socket
import threading

class DataReader(threading.Thread):
    def __init__(self, read_file, history, lock):
        super(DataReader, self).__init__()
        self.read_file = read_file
        self.history = history
        self.lock = lock

    def run(self):
        for line in self.read_file:
            # This loop will detect "shutdown()" call below
            with self.lock:
                self.history.append(line)

class PktGenRunner(object):
    def __init__(self, host, port):
        self.history = []
        self.lock = threading.Lock()
        self.start_args = ()
        self.host = host
        self.port = port
        self.read_file = None
        self.sock = None

    def read_data(self):
        for line in self.read_file:
            with self.lock:
                self.history.append(line)

    def set_args(self, *args):
        self.start_args = args

    def get_delay(self):
        with self.lock:
            values = tuple(map(float, self.history[-20:]))
            return round(sum(values) / len(values), 2)

    # See http://docs.python.org/3/reference/datamodel.html#with-statement-context-managers
    # for the context manager protocol that uses __enter__(), __exit__() methods.

    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.read_file = self.sock.makefile()
        self.reader = DataReader(self.read_file, self.history, self.lock)
        self.reader.start()
        msg = ("start:{0}\n".format(":".join(map(str, self.start_args))))
        self.sock.sendall(msg.encode())

    def __exit__(self, exc_type, exc_value, traceback):
        # Putting this code here ensures termination of pspgen
        # even if an error occurs in the script.
        self.sock.sendall("terminate\n".encode())
        self.sock.shutdown(socket.SHUT_RD)
        self.read_file.close()
        self.sock.close()
        # return True if you want to suppress the exception,
        # and you should not reraise the exception.
        return None

if __name__ == '__main__':
    # For testing...
    runner = PktGenRunner(sys.argv[1], 54321)
    runner.set_args(*sys.argv[2:])
    with runner:
        print('Waiting 10 seconds...')
        time.sleep(10)
    print('Done.')
