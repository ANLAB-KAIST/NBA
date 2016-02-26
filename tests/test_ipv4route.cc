#include <cstdint>
#include <cstdlib>
#include <cstdio>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <nba/framework/datablock.hh>
#include <nba/framework/datablock_shared.hh>
#include <nba/element/annotation.hh>
#include <nba/element/packet.hh>
#include <nba/element/packetbatch.hh>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <rte_mbuf.h>
#include "../elements/ip/ip_route_core.hh"
#include "../elements/ip/IPlookup_kernel.hh"
#include "../elements/ip/IPv4Datablocks.hh"
#if 0
#require <lib/datablock.o>
#require "../elements/ip/ip_route_core.o"
#require "../elements/ip/IPlookup_kernel.o"
#require "../elements/ip/IPv4Datablocks.o"
#endif

using namespace std;
using namespace nba;

TEST(IPLookupTest, Loading) {
    ipv4route::route_hash_t tables[33];
    ipv4route::load_rib_from_file(tables, "configs/routing_info.txt");
    size_t num_entries = 0;
    for (int i = 0; i <= 32; i++) {
        //printf("table[%d] size: %lu\n", i, tables[i].size());
        num_entries += tables[i].size();
    }
    EXPECT_EQ(282797, num_entries) << "All entries (lines) should exist.";

    // Add extra 32 entries and check overflowing.
    uint16_t *tbl24   = (uint16_t *) malloc(sizeof(uint16_t) * (ipv4route::get_TBL24_size() + 32));
    uint16_t *tbllong = (uint16_t *) malloc(sizeof(uint16_t) * (ipv4route::get_TBLlong_size() + 32));
    for (int i = 0; i < 32; i++)
        tbl24[ipv4route::get_TBL24_size() + i] = i;
    ipv4route::build_direct_fib(tables, tbl24, tbllong);
    for (int i = 0; i < 32; i++)
        EXPECT_EQ(i, tbl24[ipv4route::get_TBL24_size() + i]);
    free(tbl24);
    free(tbllong);
}

#ifdef USE_CUDA

static int getNumCUDADevices() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

class IPLookupCUDAMatchTest : public ::testing::TestWithParam<int> {
protected:
    virtual void SetUp() {
        cudaSetDevice(GetParam());
        srand(0);  // for deterministic nexthop results
        ipv4route::load_rib_from_file(tables, "configs/routing_info.txt");
        tbl24_h   = (uint16_t *) malloc(sizeof(uint16_t)
                                        * ipv4route::get_TBL24_size());
        tbllong_h = (uint16_t *) malloc(sizeof(uint16_t)
                                        * ipv4route::get_TBLlong_size());
        ipv4route::build_direct_fib(tables, tbl24_h, tbllong_h);
        cudaMalloc(&tbl24_d, sizeof(uint16_t) * ipv4route::get_TBL24_size());
        cudaMalloc(&tbllong_d, sizeof(uint16_t) * ipv4route::get_TBLlong_size());
        cudaMemcpy(tbl24_d, tbl24_h,
                   sizeof(uint16_t) * ipv4route::get_TBL24_size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(tbllong_d, tbllong_h,
                   sizeof(uint16_t) * ipv4route::get_TBL24_size(),
                   cudaMemcpyHostToDevice);
    }

    virtual void TearDown() {
        for (int i = 0; i <= 32; i++)
            tables[i].clear();
        free(tbl24_h);
        free(tbllong_h);
        cudaFree(tbl24_d);
        cudaFree(tbllong_d);
        cudaDeviceReset();
    }

    ipv4route::route_hash_t tables[33];
    uint16_t *tbl24_h;
    uint16_t *tbllong_h;
    void *tbl24_d;
    void *tbllong_d;
};

TEST_P(IPLookupCUDAMatchTest, SingleBatch) {
    void *k = ipv4_route_lookup_get_cuda_kernel();
    const char *dest_addrs[2] = { "118.223.0.3", "58.29.89.55" };
    uint16_t cpu_results[2] = { 0, 0 };

    ipv4route::direct_lookup(tbl24_h, tbllong_h,
                             ntohl(inet_addr(dest_addrs[0])), &cpu_results[0]);
    ipv4route::direct_lookup(tbl24_h, tbllong_h,
                             ntohl(inet_addr(dest_addrs[1])), &cpu_results[1]);
    EXPECT_NE(0, cpu_results[0]);
    EXPECT_NE(0, cpu_results[1]);

    const uint32_t num_batches = 1;
    const uint32_t num_pkts    = 2;
    const uint32_t count       = num_batches * num_pkts;

    struct datablock_kernel_arg *datablocks[2];
    const size_t db_arg_size = sizeof(struct datablock_kernel_arg)
                               + sizeof(struct datablock_batch_info) * num_batches;
    datablocks[0] = (struct datablock_kernel_arg *) malloc(db_arg_size);
    datablocks[1] = (struct datablock_kernel_arg *) malloc(db_arg_size);

    void *db_ipv4_dest_addrs_d     = nullptr;
    void *db_ipv4_lookup_results_d = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&db_ipv4_dest_addrs_d, db_arg_size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&db_ipv4_lookup_results_d, db_arg_size));
    ASSERT_NE(nullptr, db_ipv4_dest_addrs_d);
    ASSERT_NE(nullptr, db_ipv4_lookup_results_d);

    const size_t input_size = sizeof(uint32_t) * num_pkts;
    const size_t output_size = sizeof(uint16_t) * num_pkts;
    uint32_t *input_buffer  = (uint32_t *) malloc(input_size);
    uint16_t *output_buffer = (uint16_t *) malloc(output_size);
    ASSERT_NE(nullptr, input_buffer);
    ASSERT_NE(nullptr, output_buffer);
    input_buffer[0] = (uint32_t) inet_addr(dest_addrs[0]); // ntohl is done inside kernels
    input_buffer[1] = (uint32_t) inet_addr(dest_addrs[1]);
    output_buffer[0] = 0;
    output_buffer[1] = 0;
    void *input_buffer_d = nullptr;
    void *output_buffer_d = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&input_buffer_d, input_size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&output_buffer_d, output_size));
    ASSERT_NE(nullptr, input_buffer_d);
    ASSERT_NE(nullptr, output_buffer_d);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(input_buffer_d, input_buffer,
                                      input_size, cudaMemcpyHostToDevice));

    datablocks[0]->total_item_count_in  = num_pkts;
    datablocks[0]->total_item_count_out = 0;
    datablocks[0]->item_size_in  = sizeof(uint32_t);
    datablocks[0]->item_size_out = 0;
    datablocks[0]->batches[0].buffer_bases_in  = input_buffer_d;
    datablocks[0]->batches[0].buffer_bases_out = nullptr;
    datablocks[0]->batches[0].item_count_in  = num_pkts;
    datablocks[0]->batches[0].item_count_out = 0;
    datablocks[0]->batches[0].item_sizes_in  = nullptr;
    datablocks[0]->batches[0].item_sizes_out = nullptr;
    datablocks[0]->batches[0].item_offsets_in  = nullptr;
    datablocks[0]->batches[0].item_offsets_out = nullptr;

    datablocks[1]->total_item_count_in  = 0;
    datablocks[1]->total_item_count_out = num_pkts;
    datablocks[1]->item_size_in  = 0;
    datablocks[1]->item_size_out = sizeof(uint16_t);
    datablocks[1]->batches[0].buffer_bases_in  = nullptr;
    datablocks[1]->batches[0].buffer_bases_out = output_buffer_d;
    datablocks[1]->batches[0].item_count_in  = 0;
    datablocks[1]->batches[0].item_count_out = num_pkts;
    datablocks[1]->batches[0].item_sizes_in  = nullptr;
    datablocks[1]->batches[0].item_sizes_out = nullptr;
    datablocks[1]->batches[0].item_offsets_in  = nullptr;
    datablocks[1]->batches[0].item_offsets_out = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(db_ipv4_dest_addrs_d, datablocks[0],
                                      db_arg_size, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(db_ipv4_lookup_results_d, datablocks[1],
                                      db_arg_size, cudaMemcpyHostToDevice));
    void *dbarray_h[2] = { db_ipv4_dest_addrs_d, db_ipv4_lookup_results_d };
    void *dbarray_d = nullptr;
    uint8_t batch_ids[count] = { 0, 0 };
    uint16_t item_ids[count] = { 0, 1 };
    void *batch_ids_d = nullptr;
    void *item_ids_d  = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dbarray_d, sizeof(void*) * 2));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&batch_ids_d, sizeof(uint8_t) * count));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&item_ids_d, sizeof(uint16_t) * count));
    ASSERT_NE(nullptr, dbarray_d);
    ASSERT_NE(nullptr, batch_ids_d);
    ASSERT_NE(nullptr, item_ids_d);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dbarray_d, dbarray_h,
                                      sizeof(void*) * 2,
                                      cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(batch_ids_d, batch_ids,
                                      sizeof(uint8_t) * count,
                                      cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(item_ids_d, item_ids,
                                      sizeof(uint16_t) * count,
                                      cudaMemcpyHostToDevice));
    void *checkbits_d = nullptr;

    void *raw_args[7] = {
        &dbarray_d,
        (void *) &num_pkts,
        &batch_ids_d, &item_ids_d,
        &checkbits_d,
        &tbl24_d, &tbllong_d
    };
    ASSERT_EQ(cudaSuccess, cudaLaunchKernel(k, dim3(1), dim3(256),
              raw_args, 1024, 0));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(output_buffer, output_buffer_d,
                                      output_size, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    EXPECT_NE(0, output_buffer[0]);
    EXPECT_NE(0, output_buffer[1]);
    EXPECT_EQ(cpu_results[0], output_buffer[0]);
    EXPECT_EQ(cpu_results[1], output_buffer[1]);

    free(datablocks[0]);
    free(datablocks[1]);
    free(input_buffer);
    free(output_buffer);
    ASSERT_EQ(cudaSuccess, cudaFree(input_buffer_d));
    ASSERT_EQ(cudaSuccess, cudaFree(output_buffer_d));
    ASSERT_EQ(cudaSuccess, cudaFree(batch_ids_d));
    ASSERT_EQ(cudaSuccess, cudaFree(item_ids_d));
    ASSERT_EQ(cudaSuccess, cudaFree(db_ipv4_dest_addrs_d));
    ASSERT_EQ(cudaSuccess, cudaFree(db_ipv4_lookup_results_d));
    ASSERT_EQ(cudaSuccess, cudaFree(dbarray_d));
}

TEST_P(IPLookupCUDAMatchTest, SingleBatchWithDatablock) {
    void *k = ipv4_route_lookup_get_cuda_kernel();
    const size_t num_pkts = 2;
    const size_t pkt_size = 64;
    const char *dest_addrs[2] = { "118.223.0.3", "58.29.89.55" };

    PacketBatch *batch = new PacketBatch();
    batch->count = num_pkts;
    INIT_BATCH_MASK(batch);
    batch->banno.bitmask = 0;
    anno_set(&batch->banno, NBA_BANNO_LB_DECISION, -1);
    #if NBA_BATCHING_SCHEME == NBA_BATCHING_LINKEDLIST
    batch->first_idx = 0;
    batch->last_idx = batch->count - 1;
    batch->slot_count = batch->count;
    Packet *prev_pkt = nullptr;
    #endif
    for (unsigned pkt_idx = 0; pkt_idx < num_pkts; pkt_idx++) {
        batch->packets[pkt_idx] = (struct rte_mbuf *) malloc(pkt_size);
    }
    FOR_EACH_PACKET_ALL_INIT_PREFETCH(batch, 8u) {
        ASSERT_LT(pkt_idx, num_pkts);
        ASSERT_NE(nullptr, batch->packets[pkt_idx]);
        batch->packets[pkt_idx]->nb_segs = 1;
        batch->packets[pkt_idx]->buf_addr = (void *) ((uintptr_t) batch->packets[pkt_idx] + sizeof(struct rte_mbuf));
        batch->packets[pkt_idx]->data_off = RTE_PKTMBUF_HEADROOM;
        batch->packets[pkt_idx]->port = 0;
        batch->packets[pkt_idx]->pkt_len = pkt_size;
        batch->packets[pkt_idx]->data_len = pkt_size;
        printf("----\n");
        printf("batch pkt (mbuf) = %p\n", batch->packets[pkt_idx]);
        printf("buf_addr = %p\n", batch->packets[pkt_idx]->buf_addr);
        printf("buf_addr + sizeof(Packet) = %p\n", (uintptr_t) batch->packets[pkt_idx]->buf_addr + sizeof(Packet));
        printf("buf_addr + headroom = %p\n", (uintptr_t) batch->packets[pkt_idx]->buf_addr + RTE_PKTMBUF_HEADROOM);
        printf("mtod = %p\n", rte_pktmbuf_mtod(batch->packets[pkt_idx], char*));

        Packet *pkt = Packet::from_base_nocheck(batch->packets[pkt_idx]);
        printf("Packet = %p\n", pkt);
        new (pkt) Packet(batch, batch->packets[pkt_idx]);
        #if NBA_BATCHING_SCHEME == NBA_BATCHING_LINKEDLIST
        if (prev_pkt != nullptr) {
            prev_pkt->next_idx = pkt_idx;
            pkt->prev_idx = pkt_idx - 1;
        }
        prev_pkt = pkt;
        #endif
        //memset(pkt->data(), 0, pkt_size);
        //pkt->anno.bitmask = 0;
        //anno_set(&pkt->anno, NBA_ANNO_IFACE_IN,
        //         batch->packets[pkt_idx]->port);
        //anno_set(&pkt->anno, NBA_ANNO_TIMESTAMP, 1234);
        //anno_set(&pkt->anno, NBA_ANNO_BATCH_ID, 10000);

        // Copy the destination IP addresses
        uint32_t daddr = (uint32_t) inet_addr(dest_addrs[pkt_idx]);
        //memcpy(pkt->data() + 14 + 16, &daddr, 4);
    } END_FOR_ALL_INIT_PREFETCH;

    for (unsigned pkt_idx = 0; pkt_idx < num_pkts; pkt_idx++) {
        free(batch->packets[pkt_idx]);
    }
    delete batch;
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

INSTANTIATE_TEST_CASE_P(PerDeviceIPLookupCUDAMatchTests, IPLookupCUDAMatchTest,
                        ::testing::Values(0, getNumCUDADevices() - 1));

#endif

// vim: ts=8 sts=4 sw=4 et
