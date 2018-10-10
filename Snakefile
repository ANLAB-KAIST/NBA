#! /usr/bin/env python3
# -*- mode: python -*-
import os, sys, re
import glob
import logging
from collections import namedtuple
import subprocess
import sysconfig
from snakemake.utils import format as fmt
from snakemake.logging import logger
from distutils.version import LooseVersion

sys.path.insert(0, '.')
import compilelib


ExtLib = namedtuple('ExtLib', 'path target build_cmd clean_cmd')

def joinpath(*args):
    return os.path.normpath(os.path.join(*args))

def version():
    modern_version = LooseVersion('4.7.0')
    result = subprocess.run(['g++', '--version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True)
    for line in result.stdout.splitlines():
        m = re.search(' (\d+\.\d+(\.\d+)?)', line)
        if m:
            ver = LooseVersion(m.group(1))
            break
    else:
        raise RuntimeError('Could not detect compiler version!')
    if ver >= modern_version:
        return '-std=gnu++11'
    else:
        return '-std=c++0x'


logger.set_level(logging.DEBUG)

USE_CUDA   = bool(int(os.getenv('NBA_USE_CUDA', 1)))
USE_PHI    = bool(int(os.getenv('NBA_USE_PHI', 0)))
USE_KNAPP  = bool(int(os.getenv('NBA_USE_KNAPP', 0)))

USE_NVPROF = bool(int(os.getenv('NBA_USE_NVPROF', 0)))
USE_OPENSSL_EVP = bool(int(os.getenv('NBA_USE_OPENSSL_EVP', 1)))

NO_HUGEPAGES = bool(int(os.getenv('NBA_NO_HUGE', 0)))
# Values for batching scheme - 0: traditional, 1: continuous, 2: bitvector, 3: linkedlist
BATCHING_SCHEME   = int(os.getenv('NBA_BATCHING_SCHEME', 2))
# Values for branchpred scheme - 0: disabled, 1: enabled, 2: always
BRANCHPRED_SCHEME = int(os.getenv('NBA_BRANCHPRED_SCHEME', 0))
# Values for reuse datablocks - 0: disabled, 1: enabled
REUSE_DATABLOCKS = int(os.getenv('NBA_REUSE_DATABLOCKS', 1))
PMD      = os.getenv('NBA_PMD', 'ixgbe')
logger.debug(fmt('Compiling using {PMD} poll-mode driver...'))

# List of source and header files
THIRD_PARTY_DIR = '3rdparty'
SOURCE_DIRS = ['src/core', 'src/lib', 'elements']
SOURCE_DIRS += ['src/engines/dummy']
if USE_CUDA:
    SOURCE_DIRS += ['src/engines/cuda']
if USE_KNAPP:
    SOURCE_DIRS += ['src/engines/knapp']
    MIC_SOURCE_DIR = 'src/engines/knapp-mic'
if USE_PHI:
    SOURCE_DIRS += ['src/engines/phi']
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
if USE_KNAPP:
    MIC_SOURCE_FILES = [s for s in compilelib.find_all([MIC_SOURCE_DIR], r'^.+\.cc$')]
SOURCE_FILES.append('src/main.cc')
HEADER_FILES         = [s for s in compilelib.find_all(SOURCE_DIRS, r'^.+\.(h|hh|hpp)$') if s not in BLACKLIST]
ELEMENT_HEADER_FILES = [s for s in compilelib.find_all(['elements'], r'^.+\.(h|hh|hpp)$') if s not in BLACKLIST]

# List of object files
OBJ_DIR   = 'build'
os.makedirs(OBJ_DIR, exist_ok=True)
OBJ_FILES = [joinpath(OBJ_DIR, o) for o in map(lambda s: re.sub(r'^(.+)\.(c|cc|cpp|cu)$', r'\1.o', s), SOURCE_FILES)]
GTEST_MAIN_OBJ = 'build/src/lib/gtest/gtest_main.o'
GTEST_FUSED_OBJ = 'build/src/lib/gtest/gtest-all.o'
OBJ_FILES.remove(GTEST_MAIN_OBJ)
OBJ_FILES.remove(GTEST_FUSED_OBJ)
if USE_KNAPP:
    MIC_OBJ_DIR   = 'build/_mic'
    os.makedirs(MIC_OBJ_DIR, exist_ok=True)
    MIC_OBJ_FILES = [joinpath(MIC_OBJ_DIR, o) for o in map(lambda s: re.sub(r'^(.+)\.cc$', r'\1.o', s), MIC_SOURCE_FILES)]

# Common configurations
CXXSTD = version()

CC   = 'gcc'
CXX  = 'g++ ' + CXXSTD
if USE_KNAPP:
    MIC_CC     = 'icpc -std=c++11 -mmic -fPIC'
    MIC_CFLAGS = '-Wall -g -O2 -Iinclude -I{THIRD_PARTY_DIR}/protobuf-mic/src'
    # We link protobuf statically to make the knapp-mic binary portable.
    MIC_LIBS   = '-pthread -lscif -lrt {THIRD_PARTY_DIR}/protobuf-mic/src/.libs/libprotobuf.a'
NVCC = 'nvcc'
SUPPRESSED_CC_WARNINGS = (
    'unused-function',
    'unused-variable',
    'unused-but-set-variable',
    'unused-result',
    'unused-parameter',
)
CFLAGS      = '-march=native -O2 -g -Wall -Wextra ' + ' '.join(map(lambda s: '-Wno-' + s, SUPPRESSED_CC_WARNINGS)) + ' -Iinclude'
if os.getenv('DEBUG', 0):
    CFLAGS  = '-march=native -O0 -g3 -Wall -Wextra ' + ' '.join(map(lambda s: '-Wno-' + s, SUPPRESSED_CC_WARNINGS)) + ' -Iinclude -DDEBUG'
if os.getenv('TESTING', 0):
    CFLAGS += ' -DTESTING'

LIBS = '-pthread -lpcre -lrt'
if USE_CUDA:        CFLAGS += ' -DUSE_CUDA'
if USE_PHI:         CFLAGS += ' -DUSE_PHI'
if USE_PHI:         CFLAGS += ' -DUSE_VEC'
if USE_KNAPP:       CFLAGS += ' -DUSE_KNAPP'
if USE_KNAPP:       CFLAGS += ' -DUSE_VEC'
if USE_OPENSSL_EVP: CFLAGS += ' -DUSE_OPENSSL_EVP'
if USE_NVPROF:      CFLAGS += ' -DUSE_NVPROF'
if NO_HUGEPAGES:    CFLAGS += ' -DNBA_NO_HUGE'
CFLAGS += ' -DNBA_PMD_{0}'.format(PMD.upper())
CFLAGS += ' -DNBA_BATCHING_SCHEME={0}'.format(BATCHING_SCHEME)
CFLAGS += ' -DNBA_BRANCHPRED_SCHEME={0}'.format(BRANCHPRED_SCHEME)
CFLAGS += ' -DNBA_REUSE_DATABLOCKS={0}'.format(REUSE_DATABLOCKS)

# User-defined variables
v = os.getenv('NBA_SLEEPY_IO', 0)
if v: CFLAGS += ' -DNBA_SLEEPY_IO'
v = os.getenv('NBA_RANDOM_PORT_ACCESS', 0)
if v: CFLAGS += ' -DNBA_RANDOM_PORT_ACCESS'

# NVIDIA CUDA configurations
if USE_CUDA:
    os.makedirs('build/nvcc-temp', exist_ok=True)
    CUDA_ARCHS    = compilelib.get_cuda_arch()
    CFLAGS       += ' -I/usr/local/cuda/include'
    LIBS         += ' -L/usr/local/cuda/lib64 -lcudart' #' -lnvidia-ml'
    print(CUDA_ARCHS)
    if os.getenv('DEBUG', 0) or os.getenv('DEBUG_CUDA', 0):
        NVCFLAGS  = '-O0 -lineinfo -G -g' #' --ptxas-options=-v'
    else:
        NVCFLAGS  = '-O2 -lineinfo -g'
    NVCFLAGS     += ' -std=c++11 --keep --keep-dir build/nvcc-temp --use_fast_math --expt-relaxed-constexpr -Iinclude -I/usr/local/cuda/include'
    if len(CUDA_ARCHS) == 0:
        NVCFLAGS += ' -DMP_USE_64BIT=0' \
                  + ' -gencode arch=compute_10,code=sm_10' \
                  + ' -gencode arch=compute_12,code=sm_12' \
                  + ' -gencode arch=compute_13,code=sm_13'
        CFLAGS   += ' -DMP_USE_64BIT=0'
    else:
        NVCFLAGS += ' -DMP_USE_64BIT=1'
        CFLAGS   += ' -DMP_USE_64BIT=1'
    if 'PASCAL' in CUDA_ARCHS:
        NVCFLAGS += ' -gencode arch=compute_60,code=sm_60' \
                  + ' -gencode arch=compute_61,code=sm_61' \
                  + ' -gencode arch=compute_61,code=compute_61'
    if 'MAXWELL' in CUDA_ARCHS:
        NVCFLAGS += ' -gencode arch=compute_50,code=sm_50' \
                  + ' -gencode arch=compute_52,code=sm_52' \
                  + ' -gencode arch=compute_52,code=compute_52'
    if 'KEPLER' in CUDA_ARCHS:
        NVCFLAGS += ' -gencode arch=compute_30,code=sm_30' \
                  + ' -gencode arch=compute_35,code=sm_35' \
                  + ' -gencode arch=compute_35,code=compute_35'
    if 'FERMI' in CUDA_ARCHS:
        NVCFLAGS += ' -gencode arch=compute_20,code=sm_20' \
                  + ' -gencode arch=compute_20,code=sm_21' \
                  + ' -gencode arch=compute_20,code=compute_21'

if USE_KNAPP:
    CFLAGS += ' -I{THIRD_PARTY_DIR}/protobuf/src'
    LIBS   += ' -lscif -L{THIRD_PARTY_DIR}/protobuf/src/.libs -lprotobuf'

# NVIDIA Profiler configurations
if USE_NVPROF:
    if not USE_CUDA:
        CFLAGS += ' -I/usr/local/cuda/include'
        LIBS   += ' -L/usr/local/cuda/lib64'
    LIBS   += ' -lnvToolsExt'

# Intel Xeon Phi configurations
if USE_PHI:
    CFLAGS += ' -I/opt/intel/opencl/include'
    LIBS   += ' -L/opt/intel/opencl/lib64 -lOpenCL'

# OpenSSL configurations
SSL_PATH = os.getenv('NBA_OPENSSL_PATH') or '/usr'
CFLAGS  += ' -I{SSL_PATH}/include'
LIBS    += ' -L{SSL_PATH}/lib -lcrypto'
if USE_CUDA:
    NVCFLAGS  += fmt(' -I{SSL_PATH}/include')

# Python configurations (assuming we use the same version of Python for Snakemake and configuration scripts)
PYTHON_VERSION = '{0.major}.{0.minor}'.format(sys.version_info)
CFLAGS += ' -I{0} -fwrapv'.format(sysconfig.get_path('include'))
LIBS   += ' -L{0} -lpython{1}m {2} {3}'.format(sysconfig.get_config_var('LIBDIR'),
                                               PYTHON_VERSION,
                                               sysconfig.get_config_var('LIBS'),
                                               sysconfig.get_config_var('LINKFORSHARED'))

# click-parser configurations
CLICKPARSER_PATH = os.getenv('CLICKPARSER_PATH') or fmt('{THIRD_PARTY_DIR}/click-parser')
CFLAGS += ' -I{CLICKPARSER_PATH}/include'
LIBS   += ' -L{CLICKPARSER_PATH}/build -lclickparser'

# libev configurations
LIBEV_PREFIX = os.getenv('LIBEV_PATH', '/usr/local')
CFLAGS += ' -I{LIBEV_PREFIX}/include'
LIBS   += ' -L{LIBEV_PREFIX}/lib -lev'

# PAPI configurations
LIBS += ' -lpapi'

# DPDK configurations
DPDK_PATH = os.getenv('NBA_DPDK_PATH')
if DPDK_PATH is None:
    print('You must set NBA_DPDK_PATH environment variable.')
    sys.exit(1)
librte_pmds    = {
    'ixgbe': ['rte_pmd_ixgbe'],
    'mlx4':  ['rte_pmd_mlx4', 'rte_timer', 'ibverbs'],
    'mlnx_uio':  ['rte_pmd_mlnx_uio', 'rte_hash', 'rte_persistent'],
    'void':  ['rte_pmd_void', 'rte_kvargs'],
    'null':  ['rte_pmd_null', 'rte_kvargs'],
}
librte_names   = {'rte_ethdev', 'rte_eal', 'rte_cmdline', 'rte_sched',
                  'rte_mbuf', 'rte_mempool', 'rte_ring', 'rte_hash'}
librte_names.update(librte_pmds[PMD])
CFLAGS += ' -I{DPDK_PATH}/include'
LIBS   += ' -L{DPDK_PATH}/lib' \
          + ' -Wl,--whole-archive' \
          + ' -Wl,--start-group ' \
          + ' '.join('-l' + libname for libname in librte_names) \
          + ' -Wl,--end-group' \
          + ' -Wl,--no-whole-archive'

# Other dependencies
LIBS += ' -lnuma -ldl'

# Expand variables
CFLAGS   = fmt(CFLAGS)
CXXFLAGS = fmt(CFLAGS) + ' -Wno-literal-suffix'
LIBS     = fmt(LIBS)
if USE_KNAPP:
    MIC_CFLAGS = fmt(MIC_CFLAGS)
    MIC_LIBS   = fmt(MIC_LIBS)

# Embedded 3rd party libraries to generate rules to build them
THIRD_PARTY_LIBS = [
    ExtLib(CLICKPARSER_PATH, fmt('{CLICKPARSER_PATH}/build/libclickparser.a'),
           fmt('make -C {CLICKPARSER_PATH} -j all'),
           fmt('make -C {CLICKPARSER_PATH} clean')),
]

logger.set_level(logging.INFO)

# ---------------------------------------------------------------------------------------------

# Build rules
rule main:
    input: OBJ_FILES, [lib.target for lib in THIRD_PARTY_LIBS]
    output: 'bin/main'
    shell: '{CXX} -o {output} -Wl,--whole-archive {OBJ_FILES} -Wl,--no-whole-archive {LIBS}'

if USE_KNAPP:
    # You need to run "sudo scp knapp-mic mic0:~/" to copy to MIC.
    rule mic_main:
        input: MIC_OBJ_FILES
        output: 'bin/knapp-mic'
        shell: '{MIC_CC} -o {output} {MIC_OBJ_FILES} {MIC_LIBS}'

    rule pbgen:
        input: 'include/nba/engines/knapp/ctrl.proto'
        output: 'include/nba/engines/knapp/ctrl.pb.h', 'src/engines/knapp/ctrl.pb.cc', 'src/engines/knapp-mic/ctrl.pb.cc'
        shell: "cd include/nba/engines/knapp;" \
               "protoc --cpp_out . ctrl.proto && " \
               "sed -i 's/^#include \"ctrl.pb.h\"/#include <nba\\/engines\\/knapp\\/ctrl.pb.h>/' ctrl.pb.cc;" \
               "cp ctrl.pb.cc ../../../../src/engines/knapp;" \
               "cp ctrl.pb.cc ../../../../src/engines/knapp-mic;" \
               "rm ctrl.pb.cc"

    for srcfile in MIC_SOURCE_FILES:
        includes = [f for f in compilelib.get_includes(srcfile, 'include')]
        if srcfile.endswith('.cc'):
            objfile = re.sub(r'(.+)\.cc$', joinpath(MIC_OBJ_DIR, r'\1.o'), srcfile)
            rule:
                input: srcfile, includes
                output: objfile
                shell: '{MIC_CC} {MIC_CFLAGS} -c {input[0]} -o {output}'

for lib in THIRD_PARTY_LIBS:
    rule:
        output: lib.target
        shell: lib.build_cmd

_clean_cmds = '\n'.join(['rm -rf build bin/main bin/knapp-mic `find . -path "lib/*_map.hh"`']
                        + [lib.clean_cmd for lib in THIRD_PARTY_LIBS])
rule clean:
    shell: _clean_cmds

_test_cases, = glob_wildcards('tests/test_{case}.cc')
if not USE_CUDA:
    _test_cases.remove('cuda')
    _test_cases.remove('ipv4route')
    _test_cases.remove('ipsec')
if not USE_KNAPP:
    _test_cases.remove('knapp')
TEST_OBJ_FILES = OBJ_FILES.copy()
TEST_OBJ_FILES.append(GTEST_MAIN_OBJ)
TEST_OBJ_FILES.append(GTEST_FUSED_OBJ)
TEST_OBJ_FILES.remove('build/src/main.o')

rule cleantest:
    shell: 'rm -rf build/tests tests/test_all ' \
           + ' '.join(joinpath('tests', 'test_' + f.replace('.cc', '')) for f in _test_cases)

rule test:  # build only individual tests
    input: expand('tests/test_{case}', case=_test_cases)

rule testall:  # build a unified test suite
    input:
        testobjs=expand(joinpath(OBJ_DIR, 'tests/test_{case}.o'), case=_test_cases),
        objs=TEST_OBJ_FILES,
        libs=[lib.target for lib in THIRD_PARTY_LIBS]
    output: 'tests/test_all'
    shell: '{CXX} {CXXFLAGS} -o {output} {input.testobjs} -Wl,--whole-archive {input.objs} -Wl,--no-whole-archive {LIBS}'

for case in _test_cases:
    includes = [f for f in compilelib.get_includes(fmt('tests/test_{case}.cc'), 'include')]
    requires = [joinpath(OBJ_DIR, f) for f in compilelib.get_requires(fmt('tests/test_{case}.cc'), 'src')]
    src = fmt('tests/test_{case}.cc')
    if compilelib.has_string(src, 'int main'):
        rule:  # for individual tests
            input: src, includes, GTEST_FUSED_OBJ, req=requires
            output: fmt('tests/test_{case}')
            shell: '{CXX} {CXXFLAGS} -DTESTING -o {output} {input[0]} {input.req} {GTEST_FUSED_OBJ} {LIBS}'
        # This should be excluded from unified test because it will
        # duplicate the main() function.
    else:
        rule:  # for individual tests
            input: src, includes, GTEST_FUSED_OBJ, GTEST_MAIN_OBJ, req=requires
            output: fmt('tests/test_{case}')
            shell: '{CXX} {CXXFLAGS} -DTESTING -o {output} {input[0]} {input.req} {GTEST_FUSED_OBJ} {GTEST_MAIN_OBJ} {LIBS}'
        rule:  # for unified test suite
            input: src, includes
            output: joinpath(OBJ_DIR, fmt('tests/test_{case}.o'))
            shell: '{CXX} {CXXFLAGS} -DTESTING -o {output} -c {input[0]}'

for srcfile in SOURCE_FILES:
    # We generate build rules dynamically depending on the actual header
    # dependencies to fully exploit automatic dependency checks.
    includes = [f for f in compilelib.get_includes(srcfile, 'include')]
    if srcfile.endswith('.c'):
        objfile = re.sub(r'(.+)\.c$', joinpath(OBJ_DIR, r'\1.o'), srcfile)
        rule:
            input: srcfile, includes
            output: objfile
            shell: '{CC} {CFLAGS} -c {input[0]} -o {output}'
    elif srcfile.endswith('.cc') or srcfile.endswith('.cpp'):
        objfile = re.sub(r'(.+)\.(cc|cpp)$', joinpath(OBJ_DIR, r'\1.o'), srcfile)
        rule:
            input: srcfile, includes
            output: objfile
            shell: '{CXX} {CXXFLAGS} -c {input[0]} -o {output}'
    elif srcfile.endswith('.cu') and USE_CUDA:
        objfile = re.sub(r'(.+)\.cu$', joinpath(OBJ_DIR, r'\1.o'), srcfile)
        rule:
            input: srcfile, includes
            output: objfile
            shell: '{NVCC} {NVCFLAGS} -c {input[0]} -o {output}'

rule elemmap:
    input: ELEMENT_HEADER_FILES
    output: 'include/nba/element/element_map.hh'
    run:
        elements = ((compilelib.detect_element_def(fname), fname) for fname in ELEMENT_HEADER_FILES)
        elements = list(filter(lambda t: t[0] is not None, elements))
        with open('include/nba/element/element_map.hh', 'w') as fout:
            print('#ifndef __NBA_ELEMMAP_HH__', file=fout)
            print('#define __NBA_ELEMMAP_HH__', file=fout)
            print('/* DO NOT EDIT! This file is auto-generated. Run "snakemake elemmap" to update manually. */', file=fout)
            print('#include <unordered_map>', file=fout)
            print('#include <functional>', file=fout)
            print('#include "element.hh"', file=fout)
            for eleminfo in elements:
                print('#include "{hdrpath}"'.format(hdrpath=joinpath('../../..', eleminfo[1])), file=fout)
            print('namespace nba {', file=fout)
            print('static std::unordered_map<std::string, struct element_info> element_registry = {', file=fout)
            for idx, eleminfo in enumerate(elements):
                print('\t{{"{name}", {{ {idx}, [](void) -> Element* {{ return new {name}(); }} }} }},'
                      .format(idx=idx, name=eleminfo[0]), file=fout)
            print('};\n}\n#endif', file=fout)

# vim: ft=snakemake
