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
#include <nba/framework/test_utils.hh>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <rte_mbuf.h>
#include "../elements/ip/ip_route_core.hh"
#include "../elements/ip/IPlookup_kernel.hh"
#include "../elements/ip/IPv4Datablocks.hh"
#if 0
#require <lib/datablock.o>
#require <lib/test_utils.o>
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
    const uint32_t num_batches = 1;
    const size_t num_pkts = 2;
    const size_t pkt_size = 64;
    const char *dest_addrs[num_pkts] = { "118.223.0.3", "58.29.89.55" };
    uint16_t cpu_results[2] = { 0, 0 };

    ipv4route::direct_lookup(tbl24_h, tbllong_h,
                             ntohl(inet_addr(dest_addrs[0])), &cpu_results[0]);
    ipv4route::direct_lookup(tbl24_h, tbllong_h,
                             ntohl(inet_addr(dest_addrs[1])), &cpu_results[1]);
    EXPECT_NE(0, cpu_results[0]);
    EXPECT_NE(0, cpu_results[1]);

    PacketBatch *batch = nba::testing::create_batch(num_pkts, pkt_size,
        [&](size_t pkt_idx, struct Packet *pkt) {
            // Copy the destination IP addresses
            uint32_t daddr = (uint32_t) inet_addr(dest_addrs[pkt_idx]);
            memcpy(pkt->data() + 14 + 16, &daddr, 4);
        });
    ASSERT_NE(nullptr, batch);

    DataBlock *datablock_registry[NBA_MAX_DATABLOCKS];
    memset(datablock_registry, 0, sizeof(DataBlock*) * NBA_MAX_DATABLOCKS);
    for (unsigned dbid = 0; dbid < num_datablocks; dbid++) {
        datablock_registry[dbid] = (datablock_ctors[dbid])();
        datablock_registry[dbid]->set_id(dbid);
    }
    DataBlock *db_daddrs = datablock_registry[dbid_ipv4_dest_addrs];
    DataBlock *db_result = datablock_registry[dbid_ipv4_lookup_results];
    ASSERT_NE(nullptr, db_daddrs);
    ASSERT_NE(nullptr, db_result);

    batch->datablock_states = new struct datablock_tracker[num_datablocks];
    batch->datablock_states->aligned_item_sizes_h.ptr = malloc(sizeof(uint64_t));
    batch->datablock_states->aligned_item_sizes = (struct item_size_info *)
            batch->datablock_states->aligned_item_sizes_h.ptr;
    ASSERT_NE(nullptr, batch->datablock_states->aligned_item_sizes);
    ASSERT_EQ(cudaSuccess, cudaMalloc(&batch->datablock_states->aligned_item_sizes_d.ptr, sizeof(uint64_t)));

    size_t in_size = 0;
    size_t in_count = 0;
    struct read_roi_info rri;
    db_daddrs->get_read_roi(&rri);
    tie(in_size, in_count) = db_daddrs->calc_read_buffer_size(batch);
    ASSERT_EQ(2, in_count);
    ASSERT_EQ(rri.length * in_count, in_size);
    ASSERT_EQ(sizeof(uint32_t) * num_pkts, in_size);

    size_t out_size = 0;
    size_t out_count = 0;
    struct write_roi_info wri;
    db_result->get_write_roi(&wri);
    tie(out_size, out_count) = db_result->calc_write_buffer_size(batch);
    ASSERT_EQ(2, out_count);
    ASSERT_EQ(wri.length * out_count, out_size);
    ASSERT_EQ(sizeof(uint16_t) * num_pkts, out_size);

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

    uint32_t *input_buffer  = (uint32_t *) malloc(in_size);
    uint16_t *output_buffer = (uint16_t *) malloc(out_size);
    ASSERT_NE(nullptr, input_buffer);
    ASSERT_NE(nullptr, output_buffer);
    db_daddrs->preprocess(batch, input_buffer);
    for (unsigned i = 0; i < in_count; i++)
        ASSERT_EQ((uint32_t) inet_addr(dest_addrs[i]), input_buffer[i]);
    memset(output_buffer, 0, sizeof(uint16_t) * out_count);
    void *input_buffer_d = nullptr;
    void *output_buffer_d = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&input_buffer_d, in_size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&output_buffer_d, out_size));
    ASSERT_NE(nullptr, input_buffer_d);
    ASSERT_NE(nullptr, output_buffer_d);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(input_buffer_d, input_buffer,
                                      in_size, cudaMemcpyHostToDevice));

    datablocks[0]->total_item_count_in  = in_count;
    datablocks[0]->total_item_count_out = 0;
    datablocks[0]->item_size_in  = rri.length;
    datablocks[0]->item_size_out = 0;
    datablocks[0]->batches[0].buffer_bases_in  = input_buffer_d;
    datablocks[0]->batches[0].buffer_bases_out = nullptr;
    datablocks[0]->batches[0].item_count_in  = in_count;
    datablocks[0]->batches[0].item_count_out = 0;
    datablocks[0]->batches[0].item_sizes_in  = nullptr;
    datablocks[0]->batches[0].item_sizes_out = nullptr;
    datablocks[0]->batches[0].item_offsets_in  = nullptr;
    datablocks[0]->batches[0].item_offsets_out = nullptr;

    datablocks[1]->total_item_count_in  = 0;
    datablocks[1]->total_item_count_out = out_count;
    datablocks[1]->item_size_in  = 0;
    datablocks[1]->item_size_out = wri.length;
    datablocks[1]->batches[0].buffer_bases_in  = nullptr;
    datablocks[1]->batches[0].buffer_bases_out = output_buffer_d;
    datablocks[1]->batches[0].item_count_in  = 0;
    datablocks[1]->batches[0].item_count_out = out_count;
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
    uint8_t batch_ids[num_pkts] = { 0, 0 };
    uint16_t item_ids[num_pkts] = { 0, 1 };
    void *batch_ids_d = nullptr;
    void *item_ids_d  = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dbarray_d, sizeof(void*) * 2));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&batch_ids_d, sizeof(uint8_t) * num_pkts));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&item_ids_d, sizeof(uint16_t) * num_pkts));
    ASSERT_NE(nullptr, dbarray_d);
    ASSERT_NE(nullptr, batch_ids_d);
    ASSERT_NE(nullptr, item_ids_d);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dbarray_d, dbarray_h,
                                      sizeof(void*) * 2,
                                      cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(batch_ids_d, batch_ids,
                                      sizeof(uint8_t) * in_count,
                                      cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(item_ids_d, item_ids,
                                      sizeof(uint16_t) * in_count,
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
                                      out_size, cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    // We skip postprocessing since it has no actual elements here.
    batch->tracker.has_results = true;
    for (unsigned o = 0; o < out_count; o++) {
        EXPECT_NE(0, output_buffer[o]);
        EXPECT_EQ(cpu_results[o], output_buffer[o]);
    }

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
    ASSERT_EQ(cudaSuccess, cudaFree(batch->datablock_states->aligned_item_sizes_d.ptr));
    nba::testing::free_batch(batch);
}

INSTANTIATE_TEST_CASE_P(PerDeviceIPLookupCUDAMatchTests, IPLookupCUDAMatchTest,
                        ::testing::Values(0, getNumCUDADevices() - 1));

#endif

// vim: ts=8 sts=4 sw=4 et
