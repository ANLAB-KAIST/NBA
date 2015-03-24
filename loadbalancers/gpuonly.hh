#ifndef __LB_GPUONLY_HH__
#define __LB_GPUONLY_HH__

#include "../lib/loadbalancer.hh"

namespace nshader {

class GPUOnlyLB : public LoadBalancer
{
public:
    int gate_keeper(PacketBatch *batch, vector<ComputeDevice*>& devices)
    {
        return 0;
    }

    uint64_t update_params(SystemInspector &inspector, uint64_t timestamp)
    {
        // do nothing...
        return 1000000u;
    }
};

}

EXPORT_LOADBALANCER(GPUOnlyLB);

#endif

// vim: ts=8 sts=4 sw=4 et
