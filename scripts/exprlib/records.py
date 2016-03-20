#! /usr/bin/env python3

from abc import ABCMeta, abstractmethod
from collections import namedtuple
import re

PPCRecord = namedtuple('PPCRecord', 'node_id ppc_cpu ppc_gpu ppc_est cpu_ratio thruput')
AppThruputRecord = namedtuple('AppThrutputRecord', 'pktsz node_id conf gbps mpps')
PortThruputRecord = namedtuple('PortThruputRecord', [
    'port', 'in_pps', 'in_bps', 'out_pps', 'out_bps', 'in_errs', 'out_errs',
])
ThruputRecord = namedtuple('ThruputRecord', [
    'timestamp',
    'avg_in_pps', 'avg_in_bps', 'avg_out_pps', 'avg_out_bps', 'avg_in_errs', 'avg_out_errs',
    'total_in_pps', 'total_out_pps', 'total_in_errs', 'total_out_errs',
    'per_port_stat', 'total_fwd_pps', 'total_fwd_bps',
])


class BaseReader(metaclass=ABCMeta):

    @abstractmethod
    def parse_line(self, loop_time, line):
        raise NotImplementedError()

    @abstractmethod
    def get_records(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()


class AppThruputReader(BaseReader):

    re_total_thruput = re.compile(r'^Total forwarded pkts: (?P<Mpps>\d+\.\d+) Mpps, (?P<Gbps>\d+\.\d+) Gbps in node (?P<node_id>\d+)$')

    def __init__(self, begin_after):
        self._records = []
        self._begin_after = begin_after
        self._pktsize_hint = 0
        self._conf_hint = ''

    @property
    def pktsize_hint(self):
        return self._pktsize_hint

    @pktsize_hint.setter
    def pktsize_hint(self, value):
        self._pktsize_hint = value

    @property
    def conf_hint(self):
        return self._conf_hint

    @conf_hint.setter
    def conf_hint(self, val):
        self._conf_hint = val

    def parse_line(self, loop_time, line):
        if loop_time < self._begin_after: return
        if line.startswith('Total forwarded pkts'):
            # Parse the line like "Total forwarded pkts: xx.xx Mpps, yy.yy Gbps in node x"
            m = self.re_total_thruput.search(line)
            if m is None: return
            node_id = int(m.group('node_id'))
            self._records.append(AppThruputRecord(self._pktsize_hint, node_id, self._conf_hint,
                                                  float(m.group('Gbps')),
                                                  float(m.group('Mpps'))))

    def get_records(self):
        return self._records

    def reset(self):
        self._records.clear()


class PortThruputReader(BaseReader):

    def __init__(self):
        self._records = []

    def parse_line(self, loop_time, line):
        if line.startswith(b'port'):
            # Parse the lines like "port[x:x] x x x x .. | forwarded x.xx Mpps, y.yy Gbps"
            line = line.decode('ascii')
            m = self.rx_port_marker.search(line)
            if m is None: return
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

    def get_records(self):
        return self._records

    def reset(self):
        self._records.clear()


class PPCReader(BaseReader):

    re_num = re.compile(r'\d+')
    re_thr_gbps = re.compile(r'(\d+\.\d+) Gbps')
    re_thr_node = re.compile(r'node (\d+)$')

    # Sample lines:
    #   [PPC:1] CPU       30 GPU       47 PPC       30 CPU-Ratio 1.000
    #   Total forwarded pkts: 39.53 Mpps, 27.83 Gbps in node 1

    def __init__(self, env, prefix='PPC'):
        self._records = []
        self._last_node_thruputs = [0, 0]
        self._prefix = '[{0}:'.format(prefix)
        self._env = env

    def parse_line(self, loop_time, line):
        if line.startswith(self._prefix) and 'converge' not in line:
            pieces = tuple(s.replace(',', '') for s in line.split())
            node_id = int(self.re_num.search(pieces[0]).group(0))
            thruput = self._last_node_thruputs[node_id]
            self._records.append(PPCRecord(node_id, int(pieces[2]), int(pieces[4]), int(pieces[6]), float(pieces[8]), thruput))
        if line.startswith('Total forwarded pkts'):
            node_id = int(self.re_thr_node.search(line).group(1))
            gbps    = float(self.re_thr_gbps.search(line).group(1))
            self._last_node_thruputs[node_id] = gbps
        if line.startswith('END_OF_TEST'):
            self._env.break_main()

    def get_records(self):
        return self._records

    def reset(self):
        self._records.clear()
        self._last_node_thruputs = [0, 0]

