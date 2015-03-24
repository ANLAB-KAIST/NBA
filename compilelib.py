#! /usr/bin/env python3

import os
import re
from itertools import chain
from snakemake.io import dynamic
from snakemake.shell import shell

def joinpath(*args):
    return os.path.normpath(os.path.join(*args))

def get_cuda_arch():
    '''
    Determine currently installed NVIDIA GPU cards by PCI device ID
    and match them with the predefined GPU model lists.
    It assumes the system has only a single kind of GPUs.
    '''
    pci_list = str(shell('lspci -nn', read=True))
    supported_archs = (
        ('KEPLER_DEVICES', 'KEPLER'),
        ('FERMI_DEVICES', 'FERMI'),
    )
    for fname, devtype in supported_archs:
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                model, pciid = line.split('\t')
                pciid = pciid.replace('0x', '')
                if pciid in pci_list:
                    return devtype
    else:
        return 'OLD'

def expand_subdirs(dirlist):
    '''
    Recursively expands subdirectories for first-level subdirectory lists,
    except the root directory indicated as ".".
    '''
    for idx, dir in enumerate(dirlist[:]):
        if dir in ('.', '..'):
            continue
        for root, subdirs, files in os.walk(dir):
            for subdir in subdirs:
                dirlist.insert(idx + 1, joinpath(root, subdir))
    return dirlist

def find_all(dirlist, filepattern):
    '''
    Retrieves the list of file paths in all the given directories that matches
    with the given pattern.
    '''
    rx = re.compile(filepattern)
    results = []
    for dir in dirlist:
        for root, dirs, files in os.walk(dir):
            for fname in files:
                if rx.search(fname):
                    results.append(joinpath(root, fname))
    return results

_rx_included_local_header = re.compile(r'"(.+\.(h|hh))"')
def get_includes(srcfile, dynamic_inputs=None, visited=None):
    '''
    Gets a list of included local header files from the given source file.
    (e.g., #include "xxxx.hh")
    '''
    results = set()
    visited = visited if visited else set()
    try:
        with open(srcfile, 'r') as f:
            for line in f:
                if line.startswith('#include'):
                    m = _rx_included_local_header.search(line)
                    if not m: continue
                    p = joinpath(os.path.split(srcfile)[0], m.group(1))
                    if dynamic_inputs and any(di.endswith(p) for di in dynamic_inputs):
                        p = dynamic(p)
                    results.add(p)
        for fname in results.copy():
            if (fname.endswith('.h') or fname.endswith('.hh')) \
               and not fname in visited:
                visited.add(fname)
                results.update(s for s in get_includes(fname, dynamic_inputs, visited))
    except FileNotFoundError:
        pass
    return results

_rx_export_elem_decl = re.compile(r'^EXPORT_ELEMENT\(([a-zA-Z0-9_]+)\)')
def detect_element_def(header_file):
    with open(header_file, 'r') as fin:
        for line in fin:
            m = _rx_export_elem_decl.search(line)
            if not m: continue
            return m.group(1)

_rx_export_lb_decl = re.compile(r'EXPORT_LOADBALANCER\(([a-zA-Z0-9_]+)\)')
def detect_loadbalancer_def(header_file):
    with open(header_file, 'r') as fin:
        for line in fin:
            m = _rx_export_lb_decl.search(line)
            if not m: continue
            return m.group(1)