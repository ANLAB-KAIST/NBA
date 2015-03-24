#ifndef __NBA_ELEMENT_IPSEC_IPSECAUTHHMACSHA1_HH__
#define __NBA_ELEMENT_IPSEC_IPSECAUTHHMACSHA1_HH__

extern "C" {
#include <rte_config.h>
#include <rte_memory.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
}
#include "../../lib/element.hh"
#include "../../lib/annotation.hh"
#include "../../lib/computedevice.hh"
#include "../../lib/computecontext.hh"
#include "../../lib/nodelocalstorage.hh"
#include <vector>
#include <string>

#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <netinet/ip.h>

#include "util_esp.hh"
#include "util_ipsec_key.hh"
#include "util_sa_entry.hh"

namespace nba {

class IPsecAuthHMACSHA1 : public OffloadableElement {
public:
    IPsecAuthHMACSHA1(): OffloadableElement()
    {
        #ifdef USE_CUDA
        auto ch = [this](ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array) { this->cuda_compute_handler(ctx, res, anno_ptr_array); };
        offload_compute_handlers.insert({{"cuda", ch},});
        auto ih = [this](ComputeDevice *dev) { this->cuda_init_handler(dev); };
        offload_init_handlers.insert({{"cuda", ih},});
        #endif

        num_tunnels = 0;
        dummy_index = 0;
    }

    ~IPsecAuthHMACSHA1()
    {
    }

    const char *class_name() const { return "IPsecAuthHMACSHA1"; }
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

    void get_input_roi(struct input_roi_info *roi) const
    {
        roi->type = WHOLE_PACKET;
        roi->offset = sizeof(struct ether_hdr) + sizeof(struct iphdr);  /* input is beginning of esp-encapsulated packet */
        roi->length = -SHA_DIGEST_LENGTH;  /* Cut the trailing bytes. */
        roi->align = 0;
    }

    void get_output_roi(struct output_roi_info *roi) const
    {
        roi->type = CUSTOM_OUTPUT;
        roi->offset = 0;
        roi->length = SHA_DIGEST_LENGTH;
    }

    /* CPU-only method */
    int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno);

    /* Offloaded methods */
    #ifdef USE_CUDA
    void cuda_init_handler(ComputeDevice *device);
    void cuda_compute_handler(ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array);
    #endif
    void preproc(int input_port, void *custom_input, struct rte_mbuf *pkt, struct annotation_set *anno);
    void prepare_input(ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array);
    int postproc(int input_port, void *custom_output, struct rte_mbuf *pkt, struct annotation_set *anno);
    size_t get_desired_workgroup_size(const char *device_name) const;

protected:
    /* Maximum number of IPsec tunnels */
    int num_tunnels;
    int dummy_index;

    unordered_map<struct ipaddr_pair, int> *h_sa_table; // tunnel lookup is done in CPU only. No need for GPU ptr.
    struct hmac_sa_entry *h_key_array = NULL;       // used in CPU.
    struct hmac_sa_entry *d_key_array_ptr = NULL;   // points to the device buffer.

private:
    const int idx_pkt_offset = 0;
    const int idx_hmac_key_indice = 1;
};

EXPORT_ELEMENT(IPsecAuthHMACSHA1);

}

#endif

// vim: ts=8 sts=4 sw=4 et
