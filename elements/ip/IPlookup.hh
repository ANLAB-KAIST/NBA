#ifndef __NBA_ELEMENT_IP_IPLOOKUP_HH__
#define __NBA_ELEMENT_IP_IPLOOKUP_HH__

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

#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <assert.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <unordered_map>

#include <errno.h>

#define TBL24_SIZE  ((1 << 24) + 1)
#define TBLLONG_SIZE    ((1 << 24) + 1)

using namespace std;

namespace nba {

class IPlookup : public OffloadableElement {
protected:
    int ipv4_route_add(uint32_t addr, uint16_t len, uint16_t nexthop);
    int ipv4_route_del(uint32_t addr, uint16_t len);

    /**
     * Builds RIB from a set of IPv4 prefixes in a file.
     */
    int ipv4_load_rib_from_file(const char* filename)
    {
        FILE *fp;
        char buf[256];

        fp = fopen(filename, "r");
        if (fp == NULL) {
            getcwd(buf, 256);
            printf("NBA: IpCPULookup element: error during opening file \'%s\' from \'%s\'.: %s\n", filename, buf, strerror(errno));
        }
        assert(fp != NULL);

        while (fgets(buf, 256, fp)) {
            char *str_addr = strtok(buf, "/");
            char *str_len = strtok(NULL, "\n");
            assert(str_len != NULL);

            uint32_t addr = ntohl(inet_addr(str_addr));
            uint16_t len = atoi(str_len);

            ipv4_route_add(addr, len, rand() % 65536);
        }

        fclose(fp);

        return 0;
    }

    /**
     * Builds FIB from RIB, using DIR-24-8-BASIC scheme.
     */
    int ipv4_build_fib();

    int ipv4_get_TBL24_size()
    {
        return TBL24_SIZE;
    }

    int ipv4_get_TBLlong_size()
    {
        return TBLLONG_SIZE;
    }

    /* The CPU version. */
    void ipv4_route_lookup(
            uint32_t ip,
            uint16_t *dest)
    {
        uint16_t temp_dest;

        // TODO: make an interface to set these locks to be
        // automatically handled by process_batch() method.
        //rte_rwlock_read_lock(p_rwlock_TBL24);
        temp_dest = TBL24_h[ip >> 8];

        if (temp_dest & 0x8000u) {
            int index2 = (((uint32_t)(temp_dest & 0x7fff)) << 8) + (ip & 0xff);
            temp_dest = TBLlong_h[index2];
        }
        //rte_rwlock_read_unlock(p_rwlock_TBL24);

        *dest = temp_dest;
    }

public:
    IPlookup(): OffloadableElement()
    {
        #ifdef USE_CUDA
        auto ch = [this](ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array) { this->cuda_compute_handler(ctx, res, anno_ptr_array); };
        offload_compute_handlers.insert({{"cuda", ch},});
        auto ih = [this](ComputeDevice *dev) { this->cuda_init_handler(dev); };
        offload_init_handlers.insert({{"cuda", ih},});
        #endif

        num_tx_ports = 0;
        rr_port = 0;
        p_rwlock_TBL24 = NULL;
        p_rwlock_TBLlong = NULL;
        TBL24_h = NULL;
        TBLlong_h = NULL;
        TBL24_d = NULL;
        TBLlong_d = NULL;
    }

    virtual ~IPlookup()
    {
    }

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

    void get_input_roi(struct input_roi_info *roi) const
    {
        roi->type = PARTIAL_PACKET;
        roi->offset = 14 + 16;  /* offset of destination address */
        roi->length = 4;
        roi->align = 0;
    }

    void get_output_roi(struct output_roi_info *roi) const
    {
        roi->type = CUSTOM_OUTPUT;
        roi->offset = 0;
        roi->length = sizeof(uint16_t);
    }

    /* CPU-only method */
    int process(int input_port, struct rte_mbuf *pkt, struct annotation_set *anno);

    /* Offloaded methods */
    size_t get_desired_workgroup_size(const char *device_name) const;
    #ifdef USE_CUDA
    void cuda_init_handler(ComputeDevice *device);
    void cuda_compute_handler(ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array);
    #endif
    void preproc(int input_port, void *custom_input, struct rte_mbuf *pkt, struct annotation_set *anno);
    void prepare_input(ComputeContext *ctx, struct resource_param *res, struct annotation_set **anno_ptr_array);
    int postproc(int input_port, void *custom_output, struct rte_mbuf *pkt, struct annotation_set *anno);

protected:
    int num_tx_ports;       // Variable to store # of tx port from computation thread.
    unsigned int rr_port;   // Round-robin port #

    rte_rwlock_t    *p_rwlock_TBL24;
    rte_rwlock_t    *p_rwlock_TBLlong;

    uint16_t *TBL24_h;
    uint16_t *TBLlong_h;
    memory_t *TBL24_d;
    memory_t *TBLlong_d;
};

EXPORT_ELEMENT(IPlookup);

}

#endif

// vim: ts=8 sts=4 sw=4 et
