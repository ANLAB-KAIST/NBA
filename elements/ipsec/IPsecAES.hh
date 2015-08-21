#ifndef __NBA_ELEMENT_IPSEC_IPSECAES_HH__
#define __NBA_ELEMENT_IPSEC_IPSECAES_HH_

#include <nba/element/element.hh>
#include <vector>
#include <string>
#include <unordered_map>
#include "util_sa_entry.hh"
#include "IPsecDatablocks.hh"

namespace nba {

class IPsecAES : public OffloadableElement {
public:
    IPsecAES();
    ~IPsecAES() { }
    const char *class_name() const { return "IPsecAES"; }
    const char *port_count() const { return "1/1"; }

    int initialize();
    int initialize_global();        // per-system configuration
    int initialize_per_node();      // per-node configuration
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

    int get_offload_item_counter_dbid() const { return dbid_aes_block_info; }

    /* CPU-only method */
    int process(int input_port, Packet *pkt);

    /* Offloaded methods */
    #ifdef USE_CUDA
    void cuda_init_handler(ComputeDevice *device);
    void cuda_compute_handler(ComputeContext *ctx, struct resource_param *res);
    #endif
    int postproc(int input_port, void *custom_output, Packet *pkt);
    size_t get_desired_workgroup_size(const char *device_name) const;

    size_t get_used_datablocks(int *datablock_ids)
    {
        datablock_ids[0] = dbid_enc_payloads;
        datablock_ids[1] = dbid_iv;
        datablock_ids[2] = dbid_flow_ids;
        datablock_ids[3] = dbid_aes_block_info;
        return 4;
    }

protected:
    /* Maximum number of IPsec tunnels */
    int num_tunnels;

    /* Per-thread pointers, which points to the node local storage variables. */
    std::unordered_map<struct ipaddr_pair, int> *h_sa_table; // tunnel lookup is done in CPU only. No need for GPU ptr.
    struct aes_sa_entry *h_key_array = NULL; // used in CPU.
    memory_t d_key_array_ptr; // points to the device buffer.
};

EXPORT_ELEMENT(IPsecAES);

}

#endif

// vim: ts=8 sts=4 sw=4 et
