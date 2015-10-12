#ifndef __NBA_IPSEC_DATABLOCKS_HH__
#define __NBA_IPSEC_DATABLOCKS_HH__

#include <nba/element/packet.hh>
#include <nba/element/packetbatch.hh>
#include <nba/framework/datablock.hh>
#include <xmmintrin.h>
#include <netinet/ip.h>
#include <openssl/aes.h>
#include <openssl/sha.h>
#include "util_esp.hh"
#include "util_ipsec_key.hh"
#include "util_sa_entry.hh"
#include <rte_ether.h>

namespace nba {

extern int dbid_enc_payloads;
extern int dbid_iv;
extern int dbid_flow_ids;
extern int dbid_aes_block_info;

class IPsecEncryptedPayloadDataBlock : DataBlock
{
public:
    IPsecEncryptedPayloadDataBlock() : DataBlock()
    {}

    virtual ~IPsecEncryptedPayloadDataBlock()
    {}

    const char *name() const { return "ipsec.enc_payloads"; }

    int get_id() const { return dbid_enc_payloads; }

    void get_read_roi(struct read_roi_info *roi) const
    {
        roi->type = READ_WHOLE_PACKET;
        roi->offset = sizeof(struct ether_hdr) + sizeof(struct iphdr) + sizeof(struct esphdr);
        roi->length = 0;  /* to the end of packet */
        roi->align = CACHE_LINE_SIZE;
        roi->size_delta = SHA_DIGEST_LENGTH;
    }

    void get_write_roi(struct write_roi_info *roi) const
    {
        roi->type = WRITE_WHOLE_PACKET;
        roi->offset = sizeof(struct ether_hdr) + sizeof(struct iphdr) + sizeof(struct esphdr);
        roi->length = 0;  /* to the end of packet */
        roi->align = CACHE_LINE_SIZE;
    }
};

class IPsecIVDataBlock : DataBlock
{
public:
    IPsecIVDataBlock() : DataBlock()
    {}

    virtual ~IPsecIVDataBlock()
    {}

    const char *name() const { return "ipsec.iv"; }

    void get_read_roi(struct read_roi_info *roi) const
    {
        roi->type = READ_USER_PREPROC;
        roi->offset = 0;
        roi->length = sizeof(__m128i);
        roi->align = 0;
    }

    void get_write_roi(struct write_roi_info *roi) const
    {
        roi->type = WRITE_NONE;
        roi->offset = 0;
        roi->length = 0;
        roi->align = 0;
    }

    void calculate_read_buffer_size(PacketBatch *batch, size_t &out_bytes, size_t &out_count)
    {
        out_bytes = sizeof(__m128i) * batch->count;
        out_count = batch->count;
    }

    void preproc_batch(PacketBatch *batch, void *buffer)
    {
        char *buf = (char *) buffer;
        FOR_EACH_PACKET(batch) {
            Packet *pkt = Packet::from_base(batch->packets[pkt_idx]);
            assert(anno_isset(&pkt->anno, NBA_ANNO_IPSEC_IV1));
            assert(anno_isset(&pkt->anno, NBA_ANNO_IPSEC_IV2));
            __m128i iv = _mm_set_epi64((__m64) anno_get(&pkt->anno, NBA_ANNO_IPSEC_IV1),
                                    (__m64) anno_get(&pkt->anno, NBA_ANNO_IPSEC_IV2));
            _mm_storeu_si128((__m128i *) (buf + sizeof(__m128i) * pkt_idx), iv);
        } END_FOR;
    }
};

class IPsecFlowIDsDataBlock : DataBlock
{
public:
    IPsecFlowIDsDataBlock() : DataBlock()
    {}

    virtual ~IPsecFlowIDsDataBlock()
    {}

    const char *name() const { return "ipsec.flow_ids"; }

    void get_read_roi(struct read_roi_info *roi) const
    {
        roi->type = READ_USER_PREPROC;
        roi->offset = 0;
        roi->length = sizeof(uint64_t);
        roi->align = 0;
    }

    void get_write_roi(struct write_roi_info *roi) const
    {
        roi->type = WRITE_NONE;
        roi->offset = 0;
        roi->length = 0;
        roi->align = 0;
    }

    void *get_invalid_value() const
    {
        return (void *) &invalid_value;
    }

    void calculate_read_buffer_size(PacketBatch *batch, size_t &out_bytes, size_t &out_count)
    {
        out_bytes = sizeof(uint64_t) * batch->count;
        out_count = batch->count;
    }

    void preproc_batch(PacketBatch *batch, void *buffer)
    {
        uint64_t *buf = (uint64_t *) buffer;
        FOR_EACH_PACKET_ALL(batch) {
            if (IS_PACKET_VALID(batch, pkt_idx)) {
                Packet *pkt = Packet::from_base(batch->packets[pkt_idx]);
                assert(anno_isset(&pkt->anno, NBA_ANNO_IPSEC_FLOW_ID));
                buf[pkt_idx] = anno_get(&pkt->anno, NBA_ANNO_IPSEC_FLOW_ID);
                assert(buf[pkt_idx] < 1024);
            } else {
                // FIXME: Quick-and-dirty.. Just put invalid value in flow id to specify invalid packet.
                buf[pkt_idx] = invalid_value;
            }
        } END_FOR_ALL;
    }

    uint64_t invalid_value = 65536;
};

/*
struct aes_block_info {
    int pkt_idx;
    int block_idx;
    int pkt_offset;
};
*/

class IPsecAESBlockInfoDataBlock : DataBlock
{
public:
    IPsecAESBlockInfoDataBlock() : DataBlock(), has_pending_data(false)
    {}

    virtual ~IPsecAESBlockInfoDataBlock()
    {}

    const char *name() const { return "ipsec.aes_block_info"; }

    void get_read_roi(struct read_roi_info *roi) const
    {
        roi->type = READ_USER_PREPROC;
        roi->offset = 0;
        roi->length = sizeof(struct aes_block_info);
        roi->align = 0;
    }

    void get_write_roi(struct write_roi_info *roi) const
    {
        roi->type = WRITE_NONE;
        roi->offset = 0;
        roi->length = 0;
        roi->align = 0;
    }

    void calculate_read_buffer_size(PacketBatch *batch, size_t &out_bytes, size_t &out_count)
    {
        global_block_cnt = 0;
        unsigned global_pkt_offset = 0;
        assert(!has_pending_data);

        #ifdef DEBUG
        memset(&block_info[0], 0xcc, sizeof(struct aes_block_info) * NBA_MAX_COMP_BATCH_SIZE * (NBA_MAX_PACKET_SIZE / AES_BLOCK_SIZE));
        #endif

        FOR_EACH_PACKET(batch) {
            Packet *pkt = Packet::from_base(batch->packets[pkt_idx]);
            if (!anno_isset(&pkt->anno, NBA_ANNO_IPSEC_FLOW_ID)) {
                // h_pkt_index and h_block_offset are per-block.
                // We just skip to set them here.
                continue;
            }

            /* Per-block loop for the packet. */
            unsigned strip_hdr_len = sizeof(struct ether_hdr) + sizeof(struct iphdr) + sizeof(struct esphdr);
            unsigned pkt_len = rte_pktmbuf_data_len(batch->packets[pkt_idx]);
            unsigned payload_len = ALIGN_CEIL(pkt_len - strip_hdr_len, AES_BLOCK_SIZE);
            unsigned pkt_local_num_blocks = (payload_len + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
            for (unsigned q = 0; q < pkt_local_num_blocks; ++q) {
                unsigned global_block_idx = global_block_cnt + q;
                block_info[global_block_idx].pkt_idx = pkt_idx;
                block_info[global_block_idx].block_idx = q;
                block_info[global_block_idx].pkt_offset = global_pkt_offset;
                block_info[global_block_idx].magic = 85739;
            }
            global_block_cnt  += pkt_local_num_blocks;
            // NOTE: 여기서 align하는 부분은 DataBlock READ_WHOLE_PACKET을 프레임워크가
            //       수행할 때 align하는 것과 동일한 방법을 써야 한다.
            global_pkt_offset  = ALIGN_CEIL(global_pkt_offset + pkt_len + SHA_DIGEST_LENGTH, CACHE_LINE_SIZE);
        } END_FOR;
        out_bytes = sizeof(struct aes_block_info) * global_block_cnt;
        out_count = global_block_cnt;
        if (out_count > 0)
            has_pending_data = true;
    }

    void preproc_batch(PacketBatch *batch, void *buffer)
    {
        assert(has_pending_data);
        memcpy(buffer, &block_info[0], sizeof(struct aes_block_info) * global_block_cnt);
        has_pending_data = false;
    }

private:
    /* We can assume calculate_buffer_size() and preproc_batch() are called
     * immediately one after one, so keep the block information as
     * temporary instance member variables. */

    bool has_pending_data;

    size_t global_block_cnt;
    struct aes_block_info block_info[NBA_MAX_COMP_BATCH_SIZE * (NBA_MAX_PACKET_SIZE / AES_BLOCK_SIZE)];
};

}

#endif

// vim: ts=8 sts=4 sw=4 et
