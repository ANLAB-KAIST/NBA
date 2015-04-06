#! /usr/bin/env python3
# -*- mode: python -*-
import os, sys, re, sysconfig, glob, logging
from snakemake.utils import format as fmt
from snakemake.logging import logger
from distutils.version import LooseVersion
sys.path.insert(0, '.')
import compilelib
from pprint import pprint

logger.set_level(logging.DEBUG)

def joinpath(*args):
    return os.path.normpath(os.path.join(*args))

def version():
    modern_version = LooseVersion('4.7.0')
    for line in shell('g++ --version', iterable=True):
        m = re.search('\d\.\d\.\d', line)
        if m:
            VERSION = LooseVersion(m.group(0))
            break
    if VERSION >= modern_version:
        return '-std=gnu++11'
    else:
        return '-std=c++0x'

USE_CUDA = bool(int(os.getenv('USE_CUDA', 1)))
USE_PHI  = bool(int(os.getenv('USE_PHI', 0)))
USE_NVPROF = bool(int(os.getenv('USE_NVPROF', 0)))
USE_OPENSSL_EVP = bool(int(os.getenv('USE_OPENSSL_EVP', 1)))
NO_HUGEPAGES = bool(int(os.getenv('NBA_NO_HUGE', 0)))
PMD      = os.getenv('NBA_PMD', 'ixgbe')
logger.debug(fmt('Compiling using {PMD} poll-mode driver...'))

# List of source and header files
SOURCE_DIRS = ['lib', 'elements']
SOURCE_DIRS += ['engines/dummy']
if USE_CUDA:
    SOURCE_DIRS += ['engines/cuda']
if USE_PHI:
    SOURCE_DIRS += ['engines/phi']
BLACKLIST = {  # temporarily excluded for data-copy-optimization refactoring
    'elements/ipsec/IPsecHMACSHA1AES.cc',
    'elements/ipsec/IPsecHMACSHA1AES.hh',
    'elements/ipsec/IPsecHMACSHA1AES_kernel.cu',
    'elements/ipsec/IPsecHMACSHA1AES_kernel.hh',
    'elements/ipsec/IPsecHMACSHA1AES_kernel_core.hh',
    'elements/ipsec/IPsecAES_CBC.cc',
    'elements/ipsec/IPsecAES_CBC.hh',
    'elements/ipsec/IPsecAES_CBC_kernel.cu',
    'elements/ipsec/IPsecAES_CBC_kernel.hh',
    'elements/ipsec/IPsecAES_CBC_kernel_core.hh',
    'elements/kargus/KargusIDS_Content.cc',
    'elements/kargus/KargusIDS_Content.hh',
    'elements/kargus/KargusIDS_Content_kernel.cu',
    'elements/kargus/KargusIDS_Content_kernel.hh',
    'elements/kargus/KargusIDS_PCRE.cc',
    'elements/kargus/KargusIDS_PCRE.hh',
    'elements/kargus/KargusIDS_PCRE_kernel.cu',
    'elements/kargus/KargusIDS_PCRE_kernel.hh',
}
SOURCE_FILES = [s for s in compilelib.find_all(SOURCE_DIRS, r'^.+\.(c|cc|cpp)$') if s not in BLACKLIST]
if USE_CUDA:
    SOURCE_FILES += [s for s in compilelib.find_all(SOURCE_DIRS, r'^.+\.cu$') if s not in BLACKLIST]
SOURCE_FILES.append('main.cc')
HEADER_FILES         = [s for s in compilelib.find_all(SOURCE_DIRS, r'^.+\.(h|hh|hpp)$') if s not in BLACKLIST]
ELEMENT_HEADER_FILES = [s for s in compilelib.find_all(['elements'], r'^.+\.(h|hh|hpp)$') if s not in BLACKLIST]

# List of object files
OBJ_DIR   = 'build'
OBJ_FILES = [joinpath(OBJ_DIR, o) for o in map(lambda s: re.sub(r'^(.+)\.(c|cc|cpp|cu)$', r'\1.o', s), SOURCE_FILES)]

# Common configurations
CXXSTD = version()

CC   = 'gcc'
CXX  = 'g++ ' + CXXSTD
NVCC = 'nvcc'
SUPPRESSED_CC_WARNINGS = (
    'unused-function',
    'unused-variable',
    'unused-but-set-variable',
    'unused-result',
    'unused-parameter',
    'literal-suffix',
)
CFLAGS         = '-O2 -g -Wall -Wextra ' + ' '.join(map(lambda s: '-Wno-' + s, SUPPRESSED_CC_WARNINGS))
if os.getenv('DEBUG', 0):
    CFLAGS     = '-Og -g3 -Wall -Wextra ' + ' '.join(map(lambda s: '-Wno-' + s, SUPPRESSED_CC_WARNINGS)) + ' -DDEBUG'
#LIBS           = '-ltcmalloc_minimal -lnuma -lpthread -lpcre -lrt'
LIBS           = '-lnuma -lpthread -lpcre -lrt'
if USE_CUDA:        CFLAGS += ' -DUSE_CUDA'
if USE_PHI:         CFLAGS += ' -DUSE_PHI'
if USE_OPENSSL_EVP: CFLAGS += ' -DUSE_OPENSSL_EVP'
if USE_NVPROF:      CFLAGS += ' -DUSE_NVPROF'
if NO_HUGEPAGES:    CFLAGS += ' -DNBA_NO_HUGE'

# User-defined variables
v = os.getenv('NBA_SLEEPY_IO', 0)
if v: CFLAGS += ' -DNBA_SLEEPY_IO'
v = os.getenv('NBA_RANDOM_PORT_ACCESS', 0)
if v: CFLAGS += ' -DNBA_RANDOM_PORT_ACCESS'

# NVIDIA CUDA configurations
if USE_CUDA:
    CUDA_ARCH     = compilelib.get_cuda_arch()
    NVCFLAGS      = '-O2 -g --use_fast_math -I/usr/local/cuda/include'
    CFLAGS       += ' -I/usr/local/cuda/include'
    LIBS         += ' -L/usr/local/cuda/lib64 -lcudart' #' -lnvidia-ml'
    if os.getenv('DEBUG', 0):
        NVCFLAGS  = '-O0 --device-debug -g -G --use_fast_math -I/usr/local/cuda/include --ptxas-options=-v'
    if CUDA_ARCH == 'KEPLER':
        NVCFLAGS += ' -DMP_USE_64BIT=1 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35'
        CFLAGS   += ' -DMP_USE_64BIT=1'
    elif CUDA_ARCH == 'FERMI':
        NVCFLAGS += ' -DMP_USE_64BIT=1 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21'
        CFLAGS   += ' -DMP_USE_64BIT=1'
    else:
        NVCFLAGS +=  '-DMP_USE_64BIT=0 -gencode arch=compute_10,code=sm_10 -gencode arch=compute_12,code=sm_12 -gencode arch=compute_13,code=sm_13'
        CFLAGS   += ' -DMP_USE_64BIT=0'

# NVIDIA Profiler configurations
if USE_NVPROF:
    if not USE_CUDA:
        CFLAGS       += ' -I/usr/local/cuda/include'
        LIBS         += ' -L/usr/local/cuda/lib64'
    LIBS   += ' -lnvToolsExt'

# Intel Xeon Phi configurations
if USE_PHI:
    CFLAGS    += ' -I/opt/intel/opencl/include'
    LIBS      += ' -L/opt/intel/opencl/lib64 -lOpenCL'

# OpenSSL configurations
SSL_PATH = os.getenv('NBA_OPENSSL_PATH') or '/usr'
CFLAGS        += ' -I{SSL_PATH}/include'
LIBS          += ' -L{SSL_PATH}/lib -lcrypto'

# Python configurations (assuming we use the same version of Python for Snakemake and configuration scripts)
CFLAGS        += ' -I{0} -fwrapv'.format(sysconfig.get_path('include'))
LIBS          += ' -L{0} -lpython{1}m {2} {3}'.format(sysconfig.get_path('stdlib'),
                                                      '{0}.{1}'.format(sys.version_info.major, sys.version_info.minor),
                                                      sysconfig.get_config_var('LIBS'),
                                                      sysconfig.get_config_var('LINKFORSHARED'))

# click-parser configurations
CLICKPARSER_PATH = '$HOME/click-parser'
CFLAGS        += ' -I{CLICKPARSER_PATH}/include'
LIBS          += ' -L{CLICKPARSER_PATH}/build -lclickparser'

# libev configurations
LIBS          += ' -lev'

# DPDK configurations
DPDK_PATH = os.getenv('NBA_DPDK_PATH')
if DPDK_PATH is None:
    print('You must set NBA_DPDK_PATH environment variable.')
    sys.exit(1)
librte_pmds    = {
    'ixgbe': ['rte_pmd_ixgbe'],
    'mlx4': ['rte_pmd_mlx4', 'rte_timer'],
}
librte_names   = ['ethdev', 'rte_eal', 'rte_cmdline', 'rte_malloc', 'rte_mbuf', 'rte_mempool', 'rte_ring']
librte_names.extend(librte_pmds[PMD])
CFLAGS        += ' -I{DPDK_PATH}/include'
LIBS          += ' -L{DPDK_PATH}/lib' \
                 + ' -Wl,--whole-archive' \
                 + ' -Wl,--start-group ' \
                 + ' '.join('-l' + libname for libname in librte_names) \
                 + ' -Wl,--end-group' \
                 + ' -Wl,--no-whole-archive'

# Other dependencies
LIBS += ' -ldl'

# Expand variables
CFLAGS = fmt(CFLAGS)
LIBS = fmt(LIBS)

logger.set_level(logging.INFO)

# ---------------------------------------------------------------------------------------------

# Build rules
rule main:
    input: OBJ_FILES
    output: 'bin/main'
    shell: '{CXX} -o {output} -Wl,--whole-archive {input} -Wl,--no-whole-archive {LIBS}'

rule clean:
    shell: 'rm -rf build bin/main `find . -path "lib/*_map.hh"`'

for srcfile in SOURCE_FILES:
    # We generate build rules dynamically depending on the actual header
    # dependencies to fully exploit automatic dependency checks.
    includes = [f for f in compilelib.get_includes(srcfile)]
    if srcfile.endswith('.c'):
        outputs = re.sub(r'(.+)\.c$', joinpath(OBJ_DIR, r'\1.o'), srcfile)
        rule:
            input: srcfile, includes
            output: outputs
            shell: '{CC} {CFLAGS} -c {input[0]} -o {output}'
    elif srcfile.endswith('.cc') or srcfile.endswith('.cpp'):
        outputs = re.sub(r'(.+)\.(cc|cpp)$', joinpath(OBJ_DIR, r'\1.o'), srcfile)
        rule:
            input: srcfile, includes
            output: outputs
            shell: '{CXX} {CFLAGS} -c {input[0]} -o {output}'
    elif srcfile.endswith('.cu') and USE_CUDA:
        outputs = re.sub(r'(.+)\.cu$', joinpath(OBJ_DIR, r'\1.o'), srcfile)
        rule:
            input: srcfile, includes
            output: outputs
            shell: '{NVCC} {NVCFLAGS} -c {input[0]} -o {output}'

rule lexyacc:
    input: 'lib/configparser/nba.l', 'lib/configparser/nba.ypp'
    output: 'lib/configparser/nba.tab.cpp', 'lib/configparser/nba.tab.hpp', \
            'lib/configparser/nslex.cc', 'lib/configparser/nslex.hh'
    shell: '''
        bison -d lib/configparser/nba.ypp -b lib/configparser/nba
        flex -o lib/configparser/nslex.cc --header-file=lib/configparser/nslex.hh -Cr lib/configparser/nba.l
    '''

rule elemmap:
    input: ELEMENT_HEADER_FILES
    output: 'lib/element_map.hh'
    run:
        elements = ((compilelib.detect_element_def(fname), fname) for fname in ELEMENT_HEADER_FILES)
        elements = list(filter(lambda t: t[0] is not None, elements))
        with open('lib/element_map.hh', 'w') as fout:
            print('#ifndef __NBA_ELEMMAP_HH__', file=fout)
            print('#define __NBA_ELEMMAP_HH__', file=fout)
            print('/* DO NOT EDIT! This file is auto-generated. Run "snakemake elemmap" to update manually. */', file=fout)
            print('#include <unordered_map>', file=fout)
            print('#include <functional>', file=fout)
            print('#include "element.hh"', file=fout)
            for eleminfo in elements:
                print('#include "{hdrpath}"'.format(hdrpath=joinpath('..', eleminfo[1])), file=fout)
            print('namespace nba {', file=fout)
            print('static std::unordered_map<std::string, struct element_info> element_registry = {', file=fout)
            for idx, eleminfo in enumerate(elements):
                print('\t{{"{name}", {{ {idx}, [](void) -> Element* {{ return new {name}(); }} }} }},'
                      .format(idx=idx, name=eleminfo[0]), file=fout)
            print('};\n}\n#endif', file=fout)

# vim: ft=python
