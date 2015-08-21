#ifndef __NBA_ELEMENT_IP_IPLOOKUP_HH__
#define __NBA_ELEMENT_IP_IPLOOKUP_HH__

#include <nba/element/element.hh>
#include <vector>
#include <string>
#include <unordered_map>
#include <rte_rwlock.h>
#include "IPv4Datablocks.hh"

#define TBL24_SIZE  ((1 << 24) + 1)
#define TBLLONG_SIZE    ((1 << 24) + 1)

namespace nba {

class IPlookup : public OffloadableElement {
protected:

    /* RIB/FIB management methods. */
    int ipv4_route_add(uint32_t addr, uint16_t len, uint16_t nexthop);
    int ipv4_route_del(uint32_t addr, uint16_t len);

    /** Builds RIB from a set of IPv4 prefixes in a file. */
    int ipv4_load_rib_from_file(const char* filename);

    /** Builds FIB from RIB, using DIR-24-8-BASIC scheme. */
    int ipv4_build_fib();

    int ipv4_get_TBL24_size() { return TBL24_SIZE; }
    int ipv4_get_TBLlong_size() { return TBLLONG_SIZE; }

    /** The CPU version implementation. */
    void ipv4_route_lookup(uint32_t ip, uint16_t *dest);

public:
    IPlookup();
    virtual ~IPlookup() { }

    const char *class_name() const { return "IPlookup"; }
    const char *port_count() const { return "1/1"; }

    int initialize();
    int initialize_global();        // per-system configuration
    int initialize_per_node();  // per-node configuration
    int configure(comp_thread_context *ctx, std::vector<std::string> &args);

    void get_supported_devices(std::vector<std::string> &device_names) const
    {
        device_names.push_back("cpu");
        #ifdef USE_CUDA
        device_names.push_back("cuda");
        #endif
        #ifdef USE_PHI
        device_names.push_back("phi");
        #endif
    }

    size_t get_used_datablocks(int *datablock_ids)
    {
        datablock_ids[0] = dbid_ipv4_dest_addrs;
        datablock_ids[1] = dbid_ipv4_lookup_results;
        return 2;
    }

    /* CPU-only method */
    int process(int input_port, Packet *pkt);

    /* Offloaded methods */
    size_t get_desired_workgroup_size(const char *device_name) const;
    int get_offload_item_counter_dbid() const { return dbid_ipv4_dest_addrs; }
    #ifdef USE_CUDA
    void cuda_init_handler(ComputeDevice *device);
    void cuda_compute_handler(ComputeContext *ctx, struct resource_param *res);
    #endif
    int postproc(int input_port, void *custom_output, Packet *pkt);

protected:
    int num_tx_ports;       // Variable to store # of tx port from computation thread.
    unsigned int rr_port;   // Round-robin port #

    rte_rwlock_t *p_rwlock_TBL24;
    rte_rwlock_t *p_rwlock_TBLlong;

    uint16_t *TBL24_h;
    uint16_t *TBLlong_h;
    memory_t *TBL24_d;
    memory_t *TBLlong_d;
};

EXPORT_ELEMENT(IPlookup);

}

#endif

// vim: ts=8 sts=4 sw=4 et
