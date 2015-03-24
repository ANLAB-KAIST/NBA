#ifndef __NBA_ELEMENT_IPv6_LOOKUPIPV6ROUTE_HH__
#define __NBA_ELEMENT_IPv6_LOOKUPIPV6ROUTE_HH__


#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mbuf.h>
#include <rte_ether.h>

#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include "../../lib/computedevice.hh"
#include "../../lib/computecontext.hh"
#include "../../lib/nodelocalstorage.hh"
#include <vector>
#include <string>
#include <unordered_map>

#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <assert.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip6.h>

#include "util_routing_v6.hh"
#include <errno.h>

#ifdef USE_CUDA
#include <cuda.h>
#include "../../engines/cuda/utils.hh"
#endif

#include "IPv6Datablocks.hh"

using namespace std;

namespace nba {

class LookupIP6Route : public OffloadableElement {
public:
    LookupIP6Route(): OffloadableElement()
    {
        #ifdef USE_CUDA
        auto ch = [this](ComputeContext *ctx, struct resource_param *res) { this->cuda_compute_handler(ctx, res); };
        offload_compute_handlers.insert({{"cuda", ch},});
        auto ih = [this](ComputeDevice *dev) { this->cuda_init_handler(dev); };
        offload_init_handlers.insert({{"cuda", ih},});
        #endif

        num_tx_ports = 0;
        rr_port = 0;

        _table_ptr = NULL;
        _rwlock_ptr = NULL;
        d_tables = NULL;
        d_table_sizes = NULL;
    }

    virtual ~LookupIP6Route()
    {
    }

    const char *class_name() const { return "LookupIP6Route"; }
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
        datablock_ids[0] = dbid_ipv6_dest_addrs;
        datablock_ids[1] = dbid_ipv6_lookup_results;
        return 2;
    }

    /* CPU-only method */
    int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno);

    /* Offloaded methods */
    size_t get_desired_workgroup_size(const char *device_name) const;
    int get_offload_item_counter_dbid() const { return dbid_ipv6_dest_addrs; }
    #ifdef USE_CUDA
    void cuda_init_handler(ComputeDevice *device);
    void cuda_compute_handler(ComputeContext *ctx, struct resource_param *res);
    #endif
    void preproc(int input_port, void *custom_input, struct rte_mbuf *pkt, struct annotation_set *anno);
    void prepare_input(ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array);
    int postproc(int input_port, void *custom_output, struct rte_mbuf *pkt, struct annotation_set *anno);

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
