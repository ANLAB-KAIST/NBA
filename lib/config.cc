#include "config.hh"
#include "log.hh"
#include "strutils.hh"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>

#include <Python.h>
#include <numa.h>
#include <unistd.h>
#include <rte_config.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_log.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_memory.h>

#ifdef USE_CUDA
#include <cuda.h>
#include "../engines/cuda/utils.hh"
#endif
#ifdef USE_PHI
#include <CL/opencl.h>
#include "../engines/phi/utils.hh"
#endif

using namespace std;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"
namespace nba {

unordered_map<string, long> system_params __rte_cache_aligned;

vector<struct io_thread_conf> io_thread_confs;
vector<struct comp_thread_conf> comp_thread_confs;
vector<struct coproc_thread_conf> coproc_thread_confs;
vector<struct queue_conf> queue_confs;
unordered_map<void*, int> queue_idx_map;

bool dummy_device __rte_cache_aligned;
bool emulate_io __rte_cache_aligned;
int emulated_packet_size __rte_cache_aligned;
int emulated_ip_version __rte_cache_aligned;
int emulated_num_fixed_flows __rte_cache_aligned;
size_t num_emulated_ifaces __rte_cache_aligned;

static PyStructSequence_Field netdevice_fields[] = {
    {"device_id", "The device ID used by the underlying IO library."},
    {"driver", "Name of user-level driver used for this device."},
    {"busaddr", "PCIe bus address."},
    {"macaddr", "MAC address of device."},
    {"numa_node", "NUMA node where the device resides in."},
    {"min_rx_bufsize", "Minimum size of RX buffer."},
    {"max_rx_pktlen", "Maximum configurable length of RX packet."},
    {"max_rx_queues", "Maximum number of RX queues."},
    {"max_tx_queues", "Maximum number of TX queues."},
    {NULL, NULL}
};
static PyStructSequence_Field coprocdevice_fields[] = {
    {"device_id", "The device ID used by the device driver."},
    {"driver", "Name of the driver (e.g., cuda)"},
    {"busaddr", "PCIe bus address."},
    {"numa_node", "NUMA node where the device resides in."},
    {"name", "Model name of device."},
    {"support_concurrent_tasks", "True if supports concurrent execution of multiple tasks."},
    {"global_memory_size", "Size of device global memory."},
    // tentative
    {NULL, NULL}
};

static PyStructSequence_Desc netdevice_desc = {
    "NetDevice",
    NULL,
    netdevice_fields,
    9
};
static PyStructSequence_Desc coprocdevice_desc = {
    "CoprocDevice",
    NULL,
    coprocdevice_fields,
    7 // tentative
};

static PyTypeObject netdevice_type;
static PyTypeObject coprocdevice_type;
static PyObject *io_thread_type;
static PyObject *comp_thread_type;
static PyObject *coproc_thread_type;
static PyObject *queue_type;

static PyObject*
nba_get_netdevices(PyObject *self, PyObject *args)
{
    PyObject *plist;
    if (emulate_io) {
        plist = PyList_New(num_emulated_ifaces);
        assert(plist != NULL);
        int num_nodes = numa_num_configured_nodes();
        int cur_node = 0;
        unsigned n = 0;
        assert(num_emulated_ifaces == 1 || num_emulated_ifaces % num_nodes == 0);

        for (unsigned i = 0; i < num_emulated_ifaces; i++) {
            char tempbuf[64];
            PyObject *pnamedtuple = PyStructSequence_New(&netdevice_type);
            assert(pnamedtuple != NULL);

            PyObject *pdevid = PyLong_FromLong(i);
            PyStructSequence_SetItem(pnamedtuple, 0, pdevid);

            PyObject *pname = PyUnicode_FromString("emul_io");
            PyStructSequence_SetItem(pnamedtuple, 1, pname);

            PyOS_snprintf(tempbuf, 64, "xxxx:00:%02x.0", i);
            PyObject *pbusaddr = PyUnicode_FromString(tempbuf);
            PyStructSequence_SetItem(pnamedtuple, 2, pbusaddr);

            /* The first least significant bit should be 0 (unicast).
             * The second lsb should be 1 (locally administered).
             * "NBA" in ASCII translates to 0x4e, 0x42, 0x41,
             * and 0x4e in binary is 0b01001110.  It conforms
             * with the rule. :) */
            PyOS_snprintf(tempbuf, 64, "4e:42:41:00:00:%02X", i + 1);
            PyObject *pmacaddr = PyUnicode_FromString(tempbuf);
            PyStructSequence_SetItem(pnamedtuple, 3, pmacaddr);

            if (num_emulated_ifaces > 1 && n == num_emulated_ifaces / num_nodes) {
                cur_node ++;
                n = 0;
            }
            PyObject *pnum = PyLong_FromLong(cur_node);
            PyStructSequence_SetItem(pnamedtuple, 4, pnum);
            n++;

            pnum = PyLong_FromLong(1024L);
            PyStructSequence_SetItem(pnamedtuple, 5, pnum);
            pnum = PyLong_FromLong(1514L);
            PyStructSequence_SetItem(pnamedtuple, 6, pnum);
            pnum = PyLong_FromLong(128L);
            PyStructSequence_SetItem(pnamedtuple, 7, pnum);
            pnum = PyLong_FromLong(128L);
            PyStructSequence_SetItem(pnamedtuple, 8, pnum);

            PyList_SetItem(plist, i, pnamedtuple);
        }
    } else {
        unsigned cnt = rte_eth_dev_count();
        plist = PyList_New(cnt);
        assert(plist != NULL);

        for (unsigned i = 0; i < cnt; i++) {
            struct ether_addr macaddr;
            struct rte_eth_dev_info dev_info;
            char tempbuf[64];
            rte_eth_dev_info_get((uint8_t) i, &dev_info);
            rte_eth_macaddr_get((uint8_t) i, &macaddr);

            PyObject *pnamedtuple = PyStructSequence_New(&netdevice_type);
            assert(pnamedtuple != NULL);

            PyObject *pdevid = PyLong_FromLong(i);
            PyStructSequence_SetItem(pnamedtuple, 0, pdevid);

            PyObject *pname = PyUnicode_FromString(dev_info.driver_name);
            PyStructSequence_SetItem(pnamedtuple, 1, pname);

            PyOS_snprintf(tempbuf, 64, "%04x:%02x:%02x.%u",
                          dev_info.pci_dev->addr.domain,
                          dev_info.pci_dev->addr.bus,
                          dev_info.pci_dev->addr.devid,
                          dev_info.pci_dev->addr.function);
            PyObject *pbusaddr = PyUnicode_FromString(tempbuf);
            PyStructSequence_SetItem(pnamedtuple, 2, pbusaddr);

            PyOS_snprintf(tempbuf, 64, "%02X:%02X:%02X:%02X:%02X:%02X",
                          macaddr.addr_bytes[0], macaddr.addr_bytes[1], macaddr.addr_bytes[2],
                          macaddr.addr_bytes[3], macaddr.addr_bytes[4], macaddr.addr_bytes[5]);
            PyObject *pmacaddr = PyUnicode_FromString(tempbuf);
            PyStructSequence_SetItem(pnamedtuple, 3, pmacaddr);

            PyObject *pnum = PyLong_FromLong((long) dev_info.pci_dev->numa_node);
            PyStructSequence_SetItem(pnamedtuple, 4, pnum);
            pnum = PyLong_FromLong((long) dev_info.min_rx_bufsize);
            PyStructSequence_SetItem(pnamedtuple, 5, pnum);
            pnum = PyLong_FromLong((long) dev_info.max_rx_pktlen);
            PyStructSequence_SetItem(pnamedtuple, 6, pnum);
            pnum = PyLong_FromLong((long) dev_info.max_rx_queues);
            PyStructSequence_SetItem(pnamedtuple, 7, pnum);
            pnum = PyLong_FromLong((long) dev_info.max_tx_queues);
            PyStructSequence_SetItem(pnamedtuple, 8, pnum);

            PyList_SetItem(plist, i, pnamedtuple);
        }
    }
    return plist;
}

static PyObject*
nba_get_coprocessors(PyObject *self, PyObject *args)
{
    int count = 0;
    PyObject *plist = PyList_New(0);
    assert(plist != NULL);
#ifdef USE_CUDA
    cudaGetDeviceCount(&count); // ignore errors when GPU doesn't exist
    for (int i = 0; i < count; i ++) {
        PyObject *pnamedtuple = PyStructSequence_New(&coprocdevice_type);
        assert(pnamedtuple != NULL);

        char syspath[FILENAME_MAX], pciid[16], buf[16];
        PyObject *po;
        cudaDeviceProp prop;
        cutilSafeCall(cudaGetDeviceProperties(&prop, i));

        po = PyLong_FromLong(i);
        PyStructSequence_SetItem(pnamedtuple, 0, po);

        po = PyUnicode_FromString("cuda");
        PyStructSequence_SetItem(pnamedtuple, 1, po);

        cutilSafeCall(cudaDeviceGetPCIBusId(pciid, 16, i));
        po = PyUnicode_FromString(pciid);
        PyStructSequence_SetItem(pnamedtuple, 2, po);

        sprintf(syspath, "/sys/bus/pci/devices/%s/numa_node", pciid);
        FILE *f = fopen(syspath, "r");
        fread(buf, 16, 1, f);
        assert(feof(f));
        fclose(f);
        int numa_node = atoi(buf);
        po = PyLong_FromLong((long) numa_node);
        PyStructSequence_SetItem(pnamedtuple, 3, po);

        po = PyUnicode_FromString(prop.name);
        PyStructSequence_SetItem(pnamedtuple, 4, po);

        po = PyBool_FromLong((long) prop.concurrentKernels);
        PyStructSequence_SetItem(pnamedtuple, 5, po);

        po = PyLong_FromLong((long) prop.totalGlobalMem);
        PyStructSequence_SetItem(pnamedtuple, 6, po);

        PyList_Append(plist, pnamedtuple);
    }

    cudaDeviceReset(); // ignore errors when GPU doesn't exist
#endif
#ifdef USE_PHI
    cl_platform_id platform_ids[64];
    cl_device_id   device_ids[64];
    cl_uint num_platforms, num_devices;

    clGetPlatformIDs(64, platform_ids, &num_platforms);
    for (unsigned i = 0; i < num_platforms; i++) {
        char platform_name[256];
        size_t ret_size;
        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 256, platform_name, &ret_size);
        RTE_LOG(INFO, MAIN, "OpenCL Platform[%u]: %s\n", i, platform_name);
    }

    clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ACCELERATOR, 64, device_ids, &num_devices);
    for (unsigned i = 0; i < num_devices; i++) {
        PyObject *pnamedtuple = PyStructSequence_New(&coprocdevice_type);
        assert(pnamedtuple != NULL);

        size_t ret_size;
        PyObject *po;

        po = PyLong_FromLong(i);
        PyStructSequence_SetItem(pnamedtuple, 0, po);

        char platform_name[256];
        cl_platform_id platform_id;
        clGetDeviceInfo(device_ids[i], CL_DEVICE_PLATFORM, sizeof(platform_id), &platform_id, &ret_size);
        clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 256, platform_name, &ret_size);
        po = PyUnicode_FromString(platform_name);
        PyStructSequence_SetItem(pnamedtuple, 1, po);

        // TODO: look-up PCIe bus address using the results of lspci
        po = PyUnicode_FromString("TODO");
        PyStructSequence_SetItem(pnamedtuple, 2, po);

        // TODO: find NUMA node information from the PCIe address
        po = PyLong_FromLong(0L);
        PyStructSequence_SetItem(pnamedtuple, 3, po);

        char device_name[256];
        clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, 256, device_name, &ret_size);
        po = PyUnicode_FromString(device_name);
        PyStructSequence_SetItem(pnamedtuple, 4, po);

        cl_uint max_compute_units;
        clGetDeviceInfo(device_ids[i], CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(max_compute_units),
                        &max_compute_units, &ret_size);
        po = PyBool_FromLong(max_compute_units > 1);
        PyStructSequence_SetItem(pnamedtuple, 5, po);

        cl_ulong gmemsize;
        clGetDeviceInfo(device_ids[i], CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(gmemsize),
                        &gmemsize, &ret_size);
        po = PyLong_FromLong(gmemsize);
        PyStructSequence_SetItem(pnamedtuple, 6, po);

        PyList_Append(plist, pnamedtuple);
    }
#endif
    /* Dummy device */
    if (dummy_device) {
        int num_nodes = numa_num_configured_nodes();
        for (int i = 0; i < num_nodes; i++) {
            PyObject *pnamedtuple = PyStructSequence_New(&coprocdevice_type);
            assert(pnamedtuple != NULL);

            PyObject *po;
            char buf[16];
            po = PyUnicode_FromString("dummy");
            PyStructSequence_SetItem(pnamedtuple, 0, po);

            sprintf(buf, "xxxx:00:00.%d", i);
            po = PyUnicode_FromString(buf);
            PyStructSequence_SetItem(pnamedtuple, 1, po);

            po = PyLong_FromLong(i);
            PyStructSequence_SetItem(pnamedtuple, 2, po);

            po = PyLong_FromLong(i);
            PyStructSequence_SetItem(pnamedtuple, 3, po);

            po = PyUnicode_FromString("Dummy Computation Device");
            PyStructSequence_SetItem(pnamedtuple, 4, po);

            po = PyBool_FromLong(16L);
            PyStructSequence_SetItem(pnamedtuple, 5, po);

            po = PyLong_FromLong(1024L * 1024L * 1024L);
            PyStructSequence_SetItem(pnamedtuple, 6, po);

            PyList_Append(plist, pnamedtuple);
        }
    }

    // Sort by NUMA node
    PyObject *pGlobals = PyDict_New();
    PyObject *pLocals  = PyDict_New();
    PyObject *pkey = PyUnicode_FromString("plist");
    PyDict_SetItem(pGlobals, pkey, plist);
    PyRun_String("plist.sort(key=lambda t: t.numa_node)", Py_file_input, pGlobals, pLocals);
    Py_DECREF(pkey);
    Py_DECREF(pLocals);
    Py_DECREF(pGlobals);

    return plist;
}

static PyObject*
nba_get_cpu_node_mapping(PyObject *self, PyObject *args)
{
    int num_lcores = sysconf(_SC_NPROCESSORS_ONLN);
    int num_nodes = numa_num_configured_nodes();
    PyObject *plist = PyList_New(num_nodes);
    PyObject *psublists[NBA_MAX_NODES];
    for (int i = 0; i < num_nodes; i++) {
        psublists[i] = PyList_New(0);
        PyList_SetItem(plist, i, psublists[i]);
    }
    for (int i = 0; i < num_lcores; i++) {
        int node_id = numa_node_of_cpu(i);
        PyObject *v = PyLong_FromLong((long) i);
        PyList_Append(psublists[node_id], v);
    }
    return plist;
}

static PyMethodDef NSMethods[] = {
    {"get_netdevices", nba_get_netdevices, METH_VARARGS,
     "Retreive detailed information of the devices recognized by Intel DPDK."},
    {"get_coprocessors", nba_get_coprocessors, METH_VARARGS,
     "Retreive detailed information of the offloading devices (coprocessors) recognized by the framework."},
    {"get_cpu_node_mapping", nba_get_cpu_node_mapping, METH_VARARGS,
     "Retreive the CPU cores inside each NUMA node configured in the system as a nested list."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef NSModule = {
    PyModuleDef_HEAD_INIT, "nba", NULL, -1, NSMethods,
    NULL, NULL, NULL, NULL
};

bool check_ht_enabled()
{
    // TODO: make it portable
    char line[2048];
    unsigned len, i;
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    assert(NULL != f);
    assert(NULL != fgets(line, 2048, f));
    fclose(f);
    len = strnlen(line, 2048);
    for (i = 0; i < len; i++)
        if (line[i] == ',')
            return true;
    return false;
}

static PyObject *
nba_create_namedtuple(PyObject *namedtuple, char *name, char *fieldspec)
{
    PyObject *args = PyTuple_New(2);
    PyObject *pname = PyUnicode_FromString(name);
    PyObject *pspec = PyUnicode_FromString(fieldspec);
    PyTuple_SetItem(args, 0, pname);
    PyTuple_SetItem(args, 1, pspec);
    assert(PyCallable_Check(namedtuple));
    PyObject *ntclass = PyObject_Call(namedtuple, args, NULL);
    Py_DECREF(args);
    return ntclass;
}

PyMODINIT_FUNC
PyInit_nba(void)
{
    PyObject *mod = PyModule_Create(&NSModule);

    PyStructSequence_InitType(&netdevice_type, &netdevice_desc);
    assert(PyType_Check(&netdevice_type));
    assert(PyType_Ready(&netdevice_type) == 0);
    PyStructSequence_InitType(&coprocdevice_type, &coprocdevice_desc);
    assert(PyType_Check(&coprocdevice_type));
    assert(PyType_Ready(&coprocdevice_type) == 0);
    PyModule_AddObject(mod, "NetDevice", (PyObject *) &netdevice_type);
    PyModule_AddObject(mod, "CoprocDevice", (PyObject *) &coprocdevice_type);

    /* Create namedtuple types to represent thread/queue instances. */
    PyObject *col = PyImport_ImportModuleNoBlock("collections");
    assert(col != NULL);
    PyObject *nt = PyObject_GetAttrString(col, "namedtuple");
    PyObject *ntclass;

    ntclass = nba_create_namedtuple(nt, "IOThread", "core_id attached_rxqs mode");
    PyModule_AddObject(mod, "IOThread", ntclass);
    Py_INCREF(ntclass);
    io_thread_type = ntclass;
    
    ntclass = nba_create_namedtuple(nt, "CompThread", "core_id");
    PyModule_AddObject(mod, "CompThread", ntclass);
    Py_INCREF(ntclass);
    comp_thread_type = ntclass;

    ntclass = nba_create_namedtuple(nt, "CoprocThread", "core_id device_id");
    PyModule_AddObject(mod, "CoprocThread", ntclass);
    Py_INCREF(ntclass);
    coproc_thread_type = ntclass;

    ntclass = nba_create_namedtuple(nt, "Queue", "node_id template");
    PyModule_AddObject(mod, "Queue", ntclass);
    Py_INCREF(ntclass);
    queue_type = ntclass;

    Py_DECREF(nt);
    Py_DECREF(col);

    PyModule_AddIntConstant(mod, "num_logical_cores", sysconf(_SC_NPROCESSORS_ONLN));
    bool ht_enabled = check_ht_enabled();
    PyModule_AddIntConstant(mod, "num_physical_cores", sysconf(_SC_NPROCESSORS_ONLN) / (ht_enabled ? 2 : 1));
    PyObject *pbool = ht_enabled ? Py_True : Py_False;
    Py_INCREF(pbool);
    PyModule_AddObject(mod, "ht_enabled", pbool);
    pbool = emulate_io ? Py_True : Py_False;
    Py_INCREF(pbool);
    PyModule_AddObject(mod, "emulate_io", pbool);
    return mod;
}

static long pymap_getlong(PyObject *pmap, char *key, long default_value)
{
    long value = default_value;
    assert(PyMapping_Check(pmap));
    PyObject *_value = PyMapping_GetItemString(pmap, key);
    if (_value != NULL) {
        assert(PyLong_Check(_value));
        value = PyLong_AsLong(_value);
        Py_DECREF(_value);
    } else
        PyErr_Clear();
    return value;
}

bool load_config(const char *pyfilename)
{
    bool success = false;
    int ret = 0;
    FILE *fp = NULL;
    PyObject *p_main = NULL, *p_globals = NULL;
    PyObject *p_sys_params;
    PyObject *p_io_threads, *p_comp_threads, *p_coproc_threads;
    PyObject *p_queues, *p_thread_connections;
    int num_rxq_per_port;

    unordered_map<PyObject*, int> io_thread_idx_map;
    unordered_map<PyObject*, int> comp_thread_idx_map;
    unordered_map<PyObject*, int> coproc_thread_idx_map;
    unordered_map<int, set<PyObject*> > queue_producers;
    unordered_map<int, set<PyObject*> > queue_consumers;

    PyImport_AppendInittab("nba", &PyInit_nba);
    Py_Initialize();

    p_main = PyImport_AddModule("__main__");
    if (p_main == NULL)
        goto exit_load_config;

    fp = fopen(pyfilename, "r");
    if (fp == NULL)
        goto exit_load_config;
    ret = PyRun_SimpleFile(fp, pyfilename);
    fclose(fp);
    if (ret != 0) { // exception occurred inside the script
        /* There is no way to get exception information here,
         * but usually it would have been printed by the Python
         * interpreter during execution. */
        goto exit_load_config;
    }

    p_globals = PyModule_GetDict(p_main);
    if (p_globals == NULL)
        goto exit_load_config;

    p_sys_params = PyMapping_GetItemString(p_globals, "system_params");
    if (p_sys_params == NULL)
        goto exit_load_config;

#define LOAD_PARAM(name, defval) { \
    long val = pymap_getlong(p_sys_params, #name, defval); \
    assert(val <= NBA_MAX_ ## name); \
    system_params.insert({{#name, val}}); \
}
    LOAD_PARAM(IO_BATCH_SIZE,       64);
    LOAD_PARAM(IO_DESC_PER_HWRXQ, 1024);
    LOAD_PARAM(IO_DESC_PER_HWTXQ, 1024);

    LOAD_PARAM(COMP_BATCH_SIZE,     64);
    LOAD_PARAM(COMP_PREPKTQ_LENGTH, 32);

    LOAD_PARAM(COPROC_PPDEPTH,              64);
    LOAD_PARAM(COPROC_INPUTQ_LENGTH,        64);
    LOAD_PARAM(COPROC_COMPLETIONQ_LENGTH,   64);
    LOAD_PARAM(COPROC_CTX_PER_COMPTHREAD,    1);

    LOAD_PARAM(TASKPOOL_SIZE,  256);
    LOAD_PARAM(BATCHPOOL_SIZE, 512);
#undef LOAD_PARAM

    /* Retrieve io thread configurations. */
    p_io_threads = PyMapping_GetItemString(p_globals, "io_threads");
    if (p_io_threads == NULL)
        goto exit_load_config;
    num_rxq_per_port = 0;
    for (unsigned i = 0, len = PySequence_Size(p_io_threads);
            i < len; i ++) {
        PyObject *p_item = PySequence_GetItem(p_io_threads, i);
        assert(PyObject_IsInstance(p_item, io_thread_type)); 
        io_thread_idx_map.insert({{p_item, i}});
        struct io_thread_conf conf;

        PyObject *p_core_id = PyObject_GetAttrString(p_item, "core_id");
        conf.core_id = PyLong_AsLong(p_core_id);
        Py_DECREF(p_core_id);

        PyObject *p_rxqs = PyObject_GetAttrString(p_item, "attached_rxqs");
        assert(PySequence_Check(p_rxqs));
        for (int j = 0, len2 = PySequence_Size(p_rxqs);
                j < len2; j ++) {
            PyObject *p_rxq = PySequence_GetItem(p_rxqs, j);
            assert(PyTuple_Check(p_rxq));
            assert(PyTuple_Size(p_rxq) == 2);
            PyObject *p_num = PyTuple_GetItem(p_rxq, 0);
            int ifindex = PyLong_AsLong(p_num);
            Py_DECREF(p_num);
            p_num = PyTuple_GetItem(p_rxq, 1);
            int qidx = PyLong_AsLong(p_num);
            Py_DECREF(p_num);
            conf.attached_rxqs.push_back({ ifindex, qidx });
            Py_DECREF(p_rxq);

            if (num_rxq_per_port < qidx)
                num_rxq_per_port = qidx;
        }
        Py_DECREF(p_rxqs);

        PyObject *p_mode = PyObject_GetAttrString(p_item, "mode");
        char *mode = PyUnicode_AsUTF8(p_mode);
        if (!strcmp(mode, "normal")) {
            conf.mode = IO_NORMAL;
        } else if (!strcmp(mode, "echoback")) {
            conf.mode = IO_ECHOBACK;
        } else if (!strcmp(mode, "roundrobin")) {
            conf.mode = IO_RR;
        } else if (!strcmp(mode, "rxonly")) {
            conf.mode = IO_RXONLY;
        } else {
            assert(0); // invalid queue template name
        }
        if (emulate_io)
            conf.mode = IO_EMUL;
        Py_DECREF(p_mode);

        conf.swrxq_idx = -1;
        conf.priv = NULL;

        io_thread_confs.push_back(conf);
        Py_DECREF(p_item);
    }
    system_params.insert({{"NUM_RXQ_PER_PORT", num_rxq_per_port + 1}});

    /* Retrieve comp thread configurations. */
    p_comp_threads = PyMapping_GetItemString(p_globals, "comp_threads");
    if (p_comp_threads == NULL)
        goto exit_load_config;
    for (unsigned i = 0, len = PySequence_Size(p_comp_threads);
            i < len; i ++) {
        PyObject *p_item = PySequence_GetItem(p_comp_threads, i);
        assert(PyObject_IsInstance(p_item, comp_thread_type)); 
        comp_thread_idx_map.insert({{p_item, i}});
        struct comp_thread_conf conf;

        PyObject *p_core_id = PyObject_GetAttrString(p_item, "core_id");
        conf.core_id = PyLong_AsLong(p_core_id);
        Py_DECREF(p_core_id);

        conf.swrxq_idx = -1;
        conf.taskinq_idx = -1;
        conf.taskoutq_idx = -1;
        conf.priv = NULL;

        comp_thread_confs.push_back(conf);
        Py_DECREF(p_item);
    }

    /* Retrieve coproc thread configurations. */
    p_coproc_threads = PyMapping_GetItemString(p_globals, "coproc_threads");
    if (p_coproc_threads == NULL)
        goto exit_load_config;
    for (unsigned i = 0, len = PySequence_Size(p_coproc_threads);
            i < len; i ++) {
        PyObject *p_item = PySequence_GetItem(p_coproc_threads, i);
        assert(PyObject_IsInstance(p_item, coproc_thread_type)); 
        coproc_thread_idx_map.insert({{p_item, i}});
        struct coproc_thread_conf conf;

        PyObject *p_core_id = PyObject_GetAttrString(p_item, "core_id");
        conf.core_id = PyLong_AsLong(p_core_id);
        Py_DECREF(p_core_id);

        PyObject *p_device_id = PyObject_GetAttrString(p_item, "device_id");
        conf.device_id = PyLong_AsLong(p_device_id);
        Py_DECREF(p_device_id);

        conf.taskinq_idx = -1;
        conf.taskoutq_idx = -1;
        conf.priv = NULL;

        coproc_thread_confs.push_back(conf);
        Py_DECREF(p_item);
    }

    /* Retrieve queue configurations. */
    p_queues = PyMapping_GetItemString(p_globals, "queues");
    if (p_queues == NULL)
        goto exit_load_config;
    assert(PySequence_Check(p_queues));
    for (unsigned i = 0, len = PySequence_Size(p_queues);
            i < len; i ++) {
        PyObject *p_item = PySequence_GetItem(p_queues, i);
        assert(PyObject_IsInstance(p_item, queue_type)); 
        queue_idx_map.insert({{(void *) p_item, i}});

        struct queue_conf conf;

        PyObject *p_node_id = PyObject_GetAttrString(p_item, "node_id");
        conf.node_id = PyLong_AsLong(p_node_id);
        Py_DECREF(p_node_id);

        PyObject *p_template = PyObject_GetAttrString(p_item, "template");
        char *template_ = PyUnicode_AsUTF8(p_template);
        if (!strcmp(template_, "swrx")) {
            conf.template_ = SWRXQ;
        } else if (!strcmp(template_, "taskin")) {
            conf.template_ = TASKINQ;
        } else if (!strcmp(template_, "taskout")) {
            conf.template_ = TASKOUTQ;
        } else {
            assert(0); // invalid queue template name
        }

        conf.mp_enq = false;
        conf.mc_deq = false;
        conf.priv = NULL;

        queue_confs.push_back(conf);
        Py_DECREF(p_item);
    }
    // TODO: validate if all ports have the same number of queues.

    /* Retrieve queue connections. */
    p_thread_connections = PyMapping_GetItemString(p_globals, "thread_connections");
    if (p_thread_connections == NULL)
        goto exit_load_config;
    assert(PySequence_Check(p_thread_connections));
    for (unsigned i = 0, len = PySequence_Size(p_thread_connections);
            i < len; i ++) {
        PyObject *p_item = PySequence_GetItem(p_thread_connections, i);
        assert(PySequence_Check(p_item));
        assert(PySequence_Size(p_item) == 3);

        PyObject *p_from_thread = PySequence_GetItem(p_item, 0);
        PyObject *p_to_thread = PySequence_GetItem(p_item, 1);
        PyObject *p_queue = PySequence_GetItem(p_item, 2);

        auto ret = queue_idx_map.find((void*)p_queue);
        assert(ret != queue_idx_map.end());
        int qidx = (int) (*ret).second;

        auto ret2 = queue_producers.find(qidx);
        if (ret2 == queue_producers.end()) {
            set<PyObject *> thread_set;
            thread_set.insert(p_from_thread);
            queue_producers.insert({{qidx, thread_set}});
        } else {
            set<PyObject *> &thread_set = (*ret2).second;
            thread_set.insert(p_from_thread);
        }

        auto ret3 = queue_consumers.find(qidx);
        if (ret3 == queue_consumers.end()) {
            set<PyObject *> thread_set;
            thread_set.insert(p_to_thread);
            queue_consumers.insert({{qidx, thread_set}});
        } else {
            set<PyObject *> &thread_set = (*ret3).second;
            thread_set.insert(p_to_thread);
        }

        if (PyObject_IsInstance(p_from_thread, io_thread_type)
                && PyObject_IsInstance(p_to_thread, comp_thread_type)) {
            io_thread_confs[io_thread_idx_map[p_from_thread]].swrxq_idx = qidx;
            comp_thread_confs[comp_thread_idx_map[p_to_thread]].swrxq_idx = qidx;
        } else if (PyObject_IsInstance(p_from_thread, comp_thread_type)
                && PyObject_IsInstance(p_to_thread, coproc_thread_type)) {
            comp_thread_confs[comp_thread_idx_map[p_from_thread]].taskinq_idx = qidx;
            coproc_thread_confs[coproc_thread_idx_map[p_to_thread]].taskinq_idx = qidx;
        } else if (PyObject_IsInstance(p_from_thread, coproc_thread_type)
                && PyObject_IsInstance(p_to_thread, comp_thread_type)) {
            coproc_thread_confs[coproc_thread_idx_map[p_from_thread]].taskoutq_idx = qidx;
            comp_thread_confs[comp_thread_idx_map[p_to_thread]].taskoutq_idx = qidx;
        } else {
            assert(0); // invalid combination of connected threads
        }

        Py_DECREF(p_from_thread);
        Py_DECREF(p_to_thread);
        Py_DECREF(p_queue);
    }

    /* Update queue's MP/MC states. */
    {
        unsigned i = 0;
        for (auto it = queue_confs.begin(); it != queue_confs.end(); it++) {
            struct queue_conf &conf = *it;

            auto ret = queue_producers.find(i);
            assert(ret != queue_producers.end());
            set<PyObject *> &thread_set_p = (*ret).second;
            if (thread_set_p.size() > 1)
                conf.mp_enq = true;

            auto ret2 = queue_consumers.find(i);
            assert(ret2 != queue_consumers.end());
            set<PyObject *> &thread_set_c = (*ret2).second;
            if (thread_set_c.size() > 1)
                conf.mc_deq = true;

            i++;
        }
    }

    /* Validate that all queues are correctly set. */
    for (auto it = io_thread_confs.begin(); it != io_thread_confs.end(); it++) {
        struct io_thread_conf conf = *it;
        assert(conf.swrxq_idx != -1);
    }
    for (auto it = comp_thread_confs.begin(); it != comp_thread_confs.end(); it++) {
        struct comp_thread_conf conf = *it;
        assert(conf.swrxq_idx != -1);
        //assert(conf.taskinq_idx != -1);
        //assert(conf.taskoutq_idx != -1);
    }
    for (auto it = coproc_thread_confs.begin(); it != coproc_thread_confs.end(); it++) {
        struct coproc_thread_conf conf = *it;
        assert(conf.taskinq_idx != -1);
        assert(conf.taskoutq_idx != -1);
    }

    success = true;
exit_load_config:
    if (PyErr_Occurred())
        PyErr_Print();
    Py_Finalize();
    return success;
}

#pragma GCC diagnostic pop
}

// vim: ts=8 sts=4 sw=4 et
