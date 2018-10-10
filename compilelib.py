#! /usr/bin/env python3

from itertools import chain
import os
import re
import subprocess

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
    pci_list = str(subprocess.run(['lspci', '-nn'], stdout=subprocess.PIPE).stdout)
    supported_archs = ['MAXWELL', 'KEPLER', 'FERMI', 'PASCAL']
    devtypes_found = set()
    for devtype in supported_archs:
        fname = devtype + '_DEVICES'
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                _, pciid = line.split('\t')
                pciid = pciid.replace('0x', '')
                if pciid in pci_list:
                    devtypes_found.add(devtype)
    if len(devtypes_found) == 0:
        return []
    return list(sorted(devtypes_found, key=supported_archs.index))

def expand_subdirs(dirlist):
    '''
    Recursively expands subdirectories for first-level subdirectory lists,
    except the root directory indicated as ".".
    '''
    for idx, dir_ in enumerate(dirlist[:]):
        if dir_ in ('.', '..'):
            continue
        for root, subdirs, _ in os.walk(dir_):
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
    for root, _, files in chain(*(os.walk(d, topdown=False) for d in dirlist)):
        for fname in files:
            if rx.search(fname):
                results.append(joinpath(root, fname))
    return results

def has_string(filepath, search):
    with open(filepath, 'r') as fin:
        for line in fin:
            if isinstance(search, str) and search in line:
                return True
            elif isinstance(search, re._pattern_type) and search.search(line) is not None:
                return True
        else:
            return False

def _find_deps_with_regex(srcfile, base_dir, regexs, visited=None):
    results = set()
    visited = visited if visited else set()
    try:
        with open(srcfile, 'r', encoding='utf-8') as f:
            for line in filter(lambda l: l.startswith('#'), f):
                for regex, is_relative in regexs:
                    m = regex.search(line)
                    if not m:
                        continue
                    p = joinpath(os.path.split(srcfile)[0], m.group(1)) \
                        if is_relative \
                        else joinpath(base_dir, m.group(1))
                    results.add(p)
        for fname in results.copy():
            if not fname in visited:
                visited.add(fname)
                results.update(s for s in _find_deps_with_regex(fname, base_dir, regexs, visited))
    except FileNotFoundError:
        pass
    return results

_rx_included_local_header = re.compile(r'^#include\s*"(.+\.(h|hh))"')
_rx_included_nba_header = re.compile(r'^#include\s*<(nba/.+\.(h|hh))>')
def get_includes(srcfile, nba_include_dir):
    '''
    Gets a list of included local header files from the given source file.
    (e.g., #include <nba/xxx/xxxx.hh>)
    '''
    regexs = (
        (_rx_included_local_header, True),
        (_rx_included_nba_header, False),
    )
    return _find_deps_with_regex(srcfile, nba_include_dir, regexs)

_rx_required_local_obj_sig = re.compile(r'^#require\s*"(.+\.o)"')
_rx_required_obj_sig = re.compile(r'^#require\s*<(.+\.o)>')
def get_requires(srcfile, nba_src_dir):
    '''
    Gets a list of dependent object files from the given source file.
    (e.g., #require <lib/xxx.o>)
    '''
    regexs = (
        (_rx_required_local_obj_sig, True),
        (_rx_required_obj_sig, False),
    )
    return _find_deps_with_regex(srcfile, nba_src_dir, regexs)

_rx_export_elem_decl = re.compile(r'^EXPORT_ELEMENT\(([a-zA-Z0-9_]+)\)')
def detect_element_def(header_file):
    with open(header_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            m = _rx_export_elem_decl.search(line)
            if not m:
                continue
            return m.group(1)

_rx_export_lb_decl = re.compile(r'EXPORT_LOADBALANCER\(([a-zA-Z0-9_]+)\)')
def detect_loadbalancer_def(header_file):
    with open(header_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            m = _rx_export_lb_decl.search(line)
            if not m:
                continue
            return m.group(1)
