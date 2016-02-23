#include <nba/engines/cuda/test.hh>

using namespace std;
using namespace nba;

__global__ void noop()
{
    __syncthreads();
}

void *nba::get_test_kernel_noop()
{
    return reinterpret_cast<void *> (noop);
}

// vim: ts=8 sts=4 sw=4 et
