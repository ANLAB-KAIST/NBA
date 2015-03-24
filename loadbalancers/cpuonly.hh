#ifndef __LB_CPUONLY_HH__
#define __LB_CPUONLY_HH__

#include "../lib/loadbalancer.hh"

namespace nshader {

class CPUOnlyLB : public LoadBalancer
{
public:
    int gate_keeper(PacketBatch *batch, vector<ComputeDevice*>& devices)
    {
        return -1;
    }

    uint64_t update_params(SystemInspector &inspector, uint64_t timestamp)
    {
        // do nothing...
        return 1000000u;
    }
};

}

EXPORT_LOADBALANCER(CPUOnlyLB);

#endif

// vim: ts=8 sts=4 sw=4 et
