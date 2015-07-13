#! /usr/bin/env python3
import argparse
import sys

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
        pairs = string.split(',')
        parsed_pairs = []
        for pair in pairs:
            pieces = pair.split(':')
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
            parsed_pairs.append((host, port))
        return parsed_pairs
    return _host_port_pair_argtype

