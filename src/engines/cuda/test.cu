#include <nba/engines/cuda/test.hh>
#include <nba/core/shiftedint.hh>

using namespace std;
using namespace nba;

__global__ void noop()
{
    __syncthreads();
}

void *nba::get_test_kernel_noop()
{ return reinterpret_cast<void *> (noop); }

__global__ void shiftedint_size_check(size_t *sz_measured_in_device)
{
    *sz_measured_in_device = sizeof(nba::dev_offset_t);
}

__global__ void shiftedint_value_check
(nba::dev_offset_t *v, uint64_t *raw_v)
{
    *raw_v = v->as_value<uint64_t>();
}

void *nba::get_test_kernel_shiftedint_size_check()
{ return reinterpret_cast<void *> (shiftedint_size_check); }

void *nba::get_test_kernel_shiftedint_value_check()
{ return reinterpret_cast<void *> (shiftedint_value_check); }

// vim: ts=8 sts=4 sw=4 et
