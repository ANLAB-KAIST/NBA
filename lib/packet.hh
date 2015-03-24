#ifndef __NBA_PACKET_HH__
#define __NBA_PACKET_HH__

#include "element.hh"
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

class Packet {
public:
    Packet() : result(DROP), cloned(false) {}
    virtual ~Packet() {
        if (cloned && pkt != nullptr) {
            rte_pktmbuf_free(pkt);
        }
    }

    inline void kill() {
        result = DROP;
    }

    inline unsigned char *data() { return rte_pktmbuf_mtod(pkt, unsigned char *); }
    inline uint32_t length() { return rte_pktmbuf_data_len(pkt); }
    inline uint32_t headroom() { return rte_pktmbuf_headroom(pkt); }
    inline uint32_t tailroom() { return rte_pktmbuf_tailroom(pkt); }

    inline unsigned char *buffer() { return data() - headroom(); }
    inline unsigned char *end_buffer() { return data() + (length() + tailroom()); }
    inline uint32_t buffer_length() { return length() + headroom() + tailroom(); }

    inline bool shared() { return rte_mbuf_refcnt_read(pkt) > 1; }

    inline void pull(uint32_t len) { rte_pktmbuf_adj(pkt, (uint16_t) len); }
    inline void put(uint32_t len) { rte_pktmbuf_append(pkt, (uint16_t) len); }
    inline void take(uint32_t len) { rte_pktmbuf_trim(pkt, (uint16_t) len); }

    Packet *clone() {
        Packet *p;
        int ret = rte_mempool_get(packet_pool, (void **) &p);
        assert(ret == 0);
        struct rte_mbuf *new_mbuf = rte_pktmbuf_alloc(mbuf_pool);
        if (new_mbuf != nullptr) {
            rte_pktmbuf_attach(new_mbuf, pkt);
            p->set_mbuf_pool(mbuf_pool);
            p->set_mbuf(batch, new_mbuf);
            p->cloned = true;
            return p;
        }
        return nullptr;
    }

    Packet *uniqueify() {
        Packet *p;
        int ret = rte_mempool_get(packet_pool, (void **) &p);
        assert(ret == 0);
        struct rte_mbuf *new_mbuf = rte_pktmbuf_clone(pkt, mbuf_pool);
        if (new_mbuf != nullptr) {
            p->set_mbuf_pool(mbuf_pool);
            p->set_mbuf(batch, new_mbuf);
            p->cloned = true;
            return p;
        }
        return nullptr;
    }

    Packet *push(uint32_t len) {
        char *new_start = rte_pktmbuf_prepend(pkt, len);
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

private:
    friend class Element;

    /* @brief Set the actual packet's pktmbuf and the batch containing it. */
    void set_mbuf(PacketBatch *b, struct rte_mbuf *p) {
        batch = b;
        pkt = p;
    }

    /* @brief Set the memory pool for pktmbuf. */
    void set_mbuf_pool(struct rte_mempool *pool) {
        mbuf_pool = pool;
    }

    int get_result() {
        return result;
    }

    int result;
    PacketBatch *batch;
    struct rte_mbuf *pkt;
    struct rte_mempool *mbuf_pool;
    bool cloned;

};

struct rte_mempool *packet_create_mempool(size_t size, int node_id, int core_id);

}

#endif

// vim: ts=8 sts=4 sw=4 et
