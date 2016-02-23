#include <cstdint>
#include <cuda_runtime.h>
#include <nba/core/shiftedint.hh>
#include <nba/framework/datablock_shared.hh>
#include <gtest/gtest.h>
#include <nba/engines/cuda/test.hh>
#if 0
#require <engines/cuda/test.o>
#endif

using namespace std;
using namespace nba;

#ifdef USE_CUDA

TEST(CUDADeviceTest, Initialization) {
    EXPECT_EQ(cudaSuccess, cudaSetDevice(0));
    EXPECT_EQ(cudaSuccess, cudaDeviceReset());
}

TEST(CUDADeviceTest, NoopKernel) {
    EXPECT_EQ(cudaSuccess, cudaSetDevice(0));
    void *k = get_test_kernel_noop();
    EXPECT_EQ(cudaSuccess, cudaLaunchKernel(k, dim3(1), dim3(1), nullptr, 0, 0));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaDeviceReset());
}

#else

TEST(CUDATest, Noop) {
    EXPECT_TRUE(1);
}

#endif

// vim: ts=8 sts=4 sw=4 et
