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
    It tries to detect all GPUs and include cubins suitable for all GPU
    architectures detected.
    If your GPU is not detected correctly, update *_DEVICES
    files by referring https://pci-ids.ucw.cz/v2.2/pci.ids
    and make a pull request!
    '''
    pci_list = str(shell('lspci -nn', read=True))
    supported_archs = ['MAXWELL', 'KEPLER', 'FERMI']
    devtypes_found = set()
    for devtype in supported_archs:
        fname = devtype + '_DEVICES'
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                model, pciid = line.split('\t')
                pciid = pciid.replace('0x', '')
                if pciid in pci_list:
                    devtypes_found.add(devtype)
    if len(devtypes_found) == 0:
        return []
    return list(sorted(devtypes_found, key=lambda k: supported_archs.index(k)))

def expand_subdirs(dirlist):
    '''
    Recursively expands subdirectories for first-level subdirectory lists,
    except the root directory indicated as ".".
    '''
    for idx, dir_ in enumerate(dirlist[:]):
        if dir_ in ('.', '..'):
            continue
        for root, subdirs, files in os.walk(dir_):
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
    for root, dirs, files in chain(*map(lambda d: os.walk(d, topdown=False), dirlist)):
        for fname in files:
            if rx.search(fname):
                results.append(joinpath(root, fname))
    return results

_rx_included_local_header = re.compile(r'"(.+\.(h|hh))"')
_rx_included_nba_header = re.compile(r'<(nba/.+\.(h|hh))>')
def get_includes(srcfile, nba_include_dir, dynamic_inputs=None, visited=None):
    '''
    Gets a list of included local header files from the given source file.
    (e.g., #include <nba/xxx/xxxx.hh>)
    '''
    results = set()
    visited = visited if visited else set()
    try:
        with open(srcfile, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#include'):
                    m = _rx_included_local_header.search(line)
                    if m:
                        p = joinpath(os.path.split(srcfile)[0], m.group(1))
                        if dynamic_inputs and any(di.endswith(p) for di in dynamic_inputs):
                            p = dynamic(p)
                        results.add(p)
                    m = _rx_included_nba_header.search(line)
                    if m:
                        p = joinpath(nba_include_dir, m.group(1))
                        if dynamic_inputs and any(di.endswith(p) for di in dynamic_inputs):
                            p = dynamic(p)
                        results.add(p)
        for fname in results.copy():
            if (fname.endswith('.h') or fname.endswith('.hh')) \
               and not fname in visited:
                visited.add(fname)
                results.update(s for s in get_includes(fname, nba_include_dir,
                                                       dynamic_inputs, visited))
    except FileNotFoundError:
        pass
    return results

_rx_export_elem_decl = re.compile(r'^EXPORT_ELEMENT\(([a-zA-Z0-9_]+)\)')
def detect_element_def(header_file):
    with open(header_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            m = _rx_export_elem_decl.search(line)
            if not m: continue
            return m.group(1)

_rx_export_lb_decl = re.compile(r'EXPORT_LOADBALANCER\(([a-zA-Z0-9_]+)\)')
def detect_loadbalancer_def(header_file):
    with open(header_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            m = _rx_export_lb_decl.search(line)
            if not m: continue
            return m.group(1)
