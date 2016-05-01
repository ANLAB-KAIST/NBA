#include <nba/engines/knapp/test.hh>
#include <nba/core/shiftedint.hh>
#include <nba/framework/datablock_shared.hh>

using namespace std;
using namespace nba;

void noop()
//__global__ void noop()
{
    //__syncthreads();
}

void *nba::get_test_kernel_noop()
{ return reinterpret_cast<void *> (noop); }

void shiftedint_size_check(size_t *sz_measured_in_device)
//__global__ void shiftedint_size_check(size_t *sz_measured_in_device)
{
    *sz_measured_in_device = sizeof(nba::dev_offset_t);
}

void shiftedint_value_check
//__global__ void shiftedint_value_check
(nba::dev_offset_t *v, uint64_t *raw_v)
{
    *raw_v = v->as_value<uint64_t>();
}

void dbarg_size_check(size_t *sizes, size_t *offsets)
//__global__ void dbarg_size_check(size_t *sizes, size_t *offsets)
{
    //sizes[0] = sizeof(struct datablock_kernel_arg);
    //offsets[0] = offsetof(struct datablock_kernel_arg, batches);
    //sizes[1] = sizeof(struct datablock_batch_info);
    //offsets[1] = offsetof(struct datablock_batch_info, item_offsets);
}

void *nba::get_test_kernel_shiftedint_size_check()
{ return reinterpret_cast<void *> (shiftedint_size_check); }

void *nba::get_test_kernel_shiftedint_value_check()
{ return reinterpret_cast<void *> (shiftedint_value_check); }

void *nba::get_test_kernel_dbarg_size_check()
{ return reinterpret_cast<void *> (dbarg_size_check); }

// vim: ts=8 sts=4 sw=4 et
