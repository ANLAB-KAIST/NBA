#ifndef __NBA_ELEMENT_PKTSIZECLASSIFIER_HH__
#define __NBA_ELEMENT_PKTSIZECLASSIFIER_HH__

#include <nba/element/element.hh>
#include <vector>
#include <string>

namespace nba {

class PacketSizeClassifier : public Element {
public:
    PacketSizeClassifier() : Element()
    {
    }

    ~PacketSizeClassifier()
    {
        delete buckets;
    }

    const char *class_name() const { return "PacketSizeClassifier"; }
    const char *port_count() const { return "1/*"; }

    int initialize() {
        /* We use 16-byte segmented buckets to map output port number. */
        int i;
        buckets = new char[2048 >> 4];
        for (i = 0; i < (128 >> 4); i++)
            buckets[i] = 0;
        for (; i < (512 >> 4); i++)
            buckets[i] = 1;
        for (; i < (2048 >> 4); i++)
            buckets[i] = 2;
        return 0;
    };
    int initialize_global() { return 0; };
    int initialize_per_node() { return 0; };

    int configure(comp_thread_context *ctx, std::vector<std::string> &args)
    {
        Element::configure(ctx, args);
        // TODO: read comma-separated bucket map
        return 0;
    }

    int process(int input_port, Packet *pkt)
    {
        unsigned pkt_len = pkt->length();
        output(buckets[pkt_len >> 4]).push(pkt);
        return 0;
    }

private:
    char *buckets;
};

EXPORT_ELEMENT(PacketSizeClassifier);

}

#endif

// vim: ts=8 sts=4 sw=4 et
