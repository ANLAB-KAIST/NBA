#ifndef __NBA_ELEMENT_IPv6_LOOKUPIPV6ROUTE_HH__
#define __NBA_ELEMENT_IPv6_LOOKUPIPV6ROUTE_HH__

#include <nba/element/element.hh>
#include <vector>
#include <string>
#include <rte_rwlock.h>
#include "util_routing_v6.hh"
#include "IPv6Datablocks.hh"

namespace nba {

class LookupIP6Route : public OffloadableElement {
public:
    LookupIP6Route();
    virtual ~LookupIP6Route() { }
    const char *class_name() const { return "LookupIP6Route"; }
    const char *port_count() const { return "1/1"; }

    int initialize();
    int initialize_global();    // per-system configuration
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
        datablock_ids[0] = dbid_ipv6_dest_addrs;
        datablock_ids[1] = dbid_ipv6_lookup_results;
        return 2;
    }

    /* CPU-only method */
    int process(int input_port, Packet *pkt);

    /* Offloaded methods */
    size_t get_desired_workgroup_size(const char *device_name) const;
    int get_offload_item_counter_dbid() const { return dbid_ipv6_dest_addrs; }
    #ifdef USE_CUDA
    void cuda_init_handler(ComputeDevice *device);
    void cuda_compute_handler(ComputeContext *ctx, struct resource_param *res);
    #endif
    int postproc(int input_port, void *custom_output, Packet *pkt);

protected:
    int num_tx_ports;       // Variable to store # of tx port from computation thread.
    unsigned int rr_port;   // Round-robin port #

private:
    /* For CPU-only method */
    RoutingTableV6  _original_table;
    RoutingTableV6  *_table_ptr;
    rte_rwlock_t    *_rwlock_ptr;

    /* For offloaded methods */
    memory_t *d_tables;
    memory_t *d_table_sizes;
};

EXPORT_ELEMENT(LookupIP6Route);

}

#endif

// vim: ts=8 sts=4 sw=4 et
