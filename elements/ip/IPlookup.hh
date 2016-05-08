#ifndef __NBA_ELEMENT_IP_IPLOOKUP_HH__
#define __NBA_ELEMENT_IP_IPLOOKUP_HH__

#include <nba/element/element.hh>
#include <vector>
#include <string>
#include <unordered_map>
#include <rte_rwlock.h>
#include "ip_route_core.hh"
#include "IPv4Datablocks.hh"

namespace nba {

class IPlookup : public OffloadableElement {

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
        #ifdef USE_KNAPP
        device_names.push_back("knapp.phi");
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
    void cuda_compute_handler(ComputeDevice *dev,
                              ComputeContext *ctx,
                              struct resource_param *res);
    #endif
    #ifdef USE_KNAPP
    void knapp_init_handler(ComputeDevice *device);
    void knapp_compute_handler(ComputeDevice *dev,
                               ComputeContext *ctx,
                               struct resource_param *res);
    #endif
    int postproc(int input_port, void *custom_output, Packet *pkt);

protected:
    int num_tx_ports;       // Variable to store # of tx port from computation thread.
    unsigned int rr_port;   // Round-robin port #
    rte_rwlock_t *p_rwlock_TBL24;
    rte_rwlock_t *p_rwlock_TBLlong;
    ipv4route::route_hash_t tables[33];
    uint16_t *TBL24;
    uint16_t *TBLlong;
    host_mem_t *TBL24_h;
    host_mem_t *TBLlong_h;
    dev_mem_t *TBL24_d;
    dev_mem_t *TBLlong_d;
};

EXPORT_ELEMENT(IPlookup);

}

#endif

// vim: ts=8 sts=4 sw=4 et
