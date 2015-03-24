#ifndef __LB_WEIGHTEDRANDOM_HH__
#define __LB_WEIGHTEDRANDOM_HH__

#include "../lib/loadbalancer.hh"
#include "../lib/config.hh"

using namespace std;

namespace nba {

// WeightedRandom version 1: Just moved from old NBA's dispatcher_wr.hh
class WeightedRandomLB1 : public LoadBalancer
{
public:
    WeightedRandomLB1(): LoadBalancer() {
        cpu_count = 0;
        gpu_count = 0;
        cpu_weight = (unsigned) (100 * load_balancer_cpu_ratio);
        gpu_weight = 100 - cpu_weight;
    }
    virtual ~WeightedRandomLB1() { }

    int gate_keeper(PacketBatch *batch, vector<ComputeDevice*>& devices)
    {
        int choice = -2;    // Undefined: -2
        // TODO: this code can be used when we deal with just CPU and GPU, but what if # of devices is 3+?
        if((cpu_count * gpu_weight) < (gpu_count * cpu_weight)) {
            choice = -1;
            cpu_count ++;
            if ((cpu_count > cpu_weight) && (gpu_count > gpu_weight)) {
                cpu_count = 1;
                gpu_count = 1;
            }
        } else {
            choice = 0;
            gpu_count ++;
            if ((cpu_count > cpu_weight) && (gpu_count > gpu_weight)) {
                cpu_count = 1;
                gpu_count = 1;
            }
        }

        assert(choice != -2);
        return choice;
    }

    uint64_t update_params(SystemInspector &inspector, uint64_t timestamp)
    {
        // do nothing...
        return 1000000u;
    }

    unsigned cpu_count;
    unsigned gpu_count;
    unsigned cpu_weight;
    unsigned gpu_weight;
};

}

EXPORT_LOADBALANCER(WeightedRandomLB1);

#endif

// vim: ts=8 sts=4 sw=4 et
