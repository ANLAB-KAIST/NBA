#ifndef __NBA_PACKET_HH__
#define __NBA_PACKET_HH__

#include "element.hh"
#include <cassert>
#include <rte_eal.h>
#include <rte_memory.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>

namespace nba {

extern thread_local struct rte_mempool *packet_pool;

class PacketBatch;

/* We have to manage two memory pools:
 * first for the original memory pool that our mbuf is allocated from.
 * second for the memory pool that Packet object itself is allocated from.
 */

enum Disposition {
    /**
     * If the value >= 0, it is interpreted as output port idx.
     */
    DROP = -1,
    SLOWPATH = -2,
    PENDING = -3,
};

struct Packet {
private:
    /* Additional properties */
    #ifdef DEBUG
    uint32_t magic;
    #endif
    struct rte_mbuf *base;
    bool cloned;

    friend class Element;

public:
    struct annotation_set anno;

#define NBA_PACKET_MAGIC 0x392cafcdu

public:
    /**
     * Get the pointer to the beginning of Packet object
     * from the "base" IO layer packet objects.
     * This "nocheck" version is used only for initialization.
     */
    static inline Packet *from_base_nocheck(void *base) {
        if (base == nullptr) return nullptr;
        return reinterpret_cast<Packet *>((char *) base + sizeof(struct rte_mbuf));
    }

    /**
     * Get the pointer to the beginning of Packet object
     * from the "base" IO layer packet objects.
     * In debugging mode, it performs additinoal integrity check using a
     * magic number placed in the beginning of Packet struct.
     */
    static inline Packet *from_base(void *base) {
        if (base == nullptr) return nullptr;
        #ifdef DEBUG
        assert(NBA_PACKET_MAGIC == *(uint32_t*) ((char *) base + sizeof(struct rte_mbuf)));
        #endif
        return reinterpret_cast<Packet *>((char *) base + sizeof(struct rte_mbuf));
    }

    /**
     * Initialize the Packet object.
     */
    Packet(PacketBatch *mother, void *base) :
    #ifdef DEBUG
    magic(NBA_PACKET_MAGIC),
    #endif
    base((struct rte_mbuf *) base), cloned(false)
    { }

    ~Packet() {
        if (cloned && base != nullptr) {
            rte_pktmbuf_free(base);
        }
    }

    inline void kill() { /** Deprecated. Use "return DROP" instead. */ }

    inline unsigned char *data() { return rte_pktmbuf_mtod(base, unsigned char *); }
    inline uint32_t length() { return rte_pktmbuf_data_len(base); }
    inline uint32_t headroom() { return rte_pktmbuf_headroom(base); }
    inline uint32_t tailroom() { return rte_pktmbuf_tailroom(base); }

    inline unsigned char *buffer() { return data() - headroom(); }
    inline unsigned char *end_buffer() { return data() + (length() + tailroom()); }
    inline uint32_t buffer_length() { return length() + headroom() + tailroom(); }

    inline bool shared() { return rte_mbuf_refcnt_read(base) > 1; }

    inline void pull(uint32_t len) { rte_pktmbuf_adj(base, (uint16_t) len); }
    inline void put(uint32_t len) { rte_pktmbuf_append(base, (uint16_t) len); }
    inline void take(uint32_t len) { rte_pktmbuf_trim(base, (uint16_t) len); }

    Packet *clone() {
        //Packet *p;
        //int ret = rte_mempool_get(packet_pool, (void **) &p);
        //assert(ret == 0);
        //struct rte_mbuf *new_mbuf = rte_pktmbuf_alloc(mbuf_pool);
        //if (new_mbuf != nullptr) {
        //    rte_pktmbuf_attach(new_mbuf, base);
        //    p->set_mbuf_pool(mbuf_pool);
        //    p->set_mbuf(batch, new_mbuf);
        //    p->cloned = true;
        //    return p;
        //}
        return nullptr;
    }

    Packet *uniqueify() {
        //Packet *p;
        //int ret = rte_mempool_get(packet_pool, (void **) &p);
        //assert(ret == 0);
        //struct rte_mbuf *new_mbuf = rte_pktmbuf_clone(base, mbuf_pool);
        //if (new_mbuf != nullptr) {
        //    p->set_mbuf_pool(mbuf_pool);
        //    p->set_mbuf(batch, new_mbuf);
        //    p->cloned = true;
        //    return p;
        //}
        return nullptr;
    }

    Packet *push(uint32_t len) {
        char *new_start = rte_pktmbuf_prepend(base, len);
        if (new_start != nullptr)
            return this;
        return nullptr;
    }

#if 0
    Packet *push_mac_header(uint32_t len);

    inline bool has_mac_header();
    inline unsigned char *mac_header();
    inline int mac_header_offset();
    inline uint32_t mac_header_length();
    inline int mac_length();
    inline void set_mac_header(unsigned char *p);
    inline void set_mac_header(unsigned char *p, uint32_t len);
    inline void clear_mac_header();

    inline bool has_network_header();
    inline unsigned char *network_header();
    inline int network_header_offset();
    inline uint32_t network_header_length();
    inline int network_length();
    inline void set_network_header(unsigned char *p, uint32_t len);
    inline void set_network_header_length(uint32_t len);
    inline void clear_network_header();

    inline bool has_transport_header();
    inline unsigned char *transport_header();
    inline int transport_header_offset();
    inline int transport_length();
    inline void clear_transport_header();

    inline click_ether *ether_header();
    inline void set_ether_header(click_ether *ethh);

    inline click_ip *ip_header();
    inline int ip_header_offset();
    inline uint32_t ip_header_length();
    inline void set_ip_header(click_ip *iph, uint32_t len);

    inline click_ip6 *ip6_header();
    inline int ip6_header_offset();
    inline uint32_t ip6_header_length();
    inline void set_ip6_header(click_ip6 *ip6h);
    inline void set_ip6_header(click_ip6 *ip6h, uint32_t len);

    inline click_icmp *icmp_header();
    inline click_tcp *tcp_header();
    inline click_udp *udp_header();

    inline const Timestamp &timestamp_anno();
    inline Timestamp &timestamp_anno();
    inline void set_timestamp_anno(const Timestamp &t);

    inline net_device *device_anno();
    inline void set_device_anno(net_device *dev);

    /** @brief Values for packet_type_anno().
     * Must agree with Linux's PACKET_ constants in <linux/if_packet.h>. */
    enum PacketType {
	HOST = 0,		/**< Packet was sent to this host. */
	BROADCAST = 1,		/**< Packet was sent to a link-level multicast
				     address. */
	MULTICAST = 2,		/**< Packet was sent to a link-level multicast
				     address. */
	OTHERHOST = 3,		/**< Packet was sent to a different host, but
				     received anyway.  The receiving device is
				     probably in promiscuous mode. */
	OUTGOING = 4,		/**< Packet was generated by this host and is
				     being sent elsewhere. */
	LOOPBACK = 5,
	FASTROUTE = 6
    };
    inline PacketType packet_type_anno();
    inline void set_packet_type_anno(PacketType t);
#endif

    // TODO: a lot of annotation-related methods...
};

struct rte_mempool *packet_create_mempool(size_t size, int node_id, int core_id);

}

#endif

// vim: ts=8 sts=4 sw=4 et
