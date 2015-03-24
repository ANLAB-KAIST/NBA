#include "IPsecAES_CBC_kernel_core.hh"
#include "IPsecAES_CBC_kernel.hh"

#include <cuda.h>
#include "../../engines/cuda/utils.hh"

#include <stdint.h>

#include <assert.h>
#include <stdio.h>


/*******************************************************************
  AES CBC kernel
******************************************************************/

/* former prototype
__global__ void
AES_cbc_128_encrypt_kernel_SharedMem_cbc(const uint8_t       *in_all,
				     uint8_t             *out_all,
				     const uint32_t      *pkt_offset,
				     const uint8_t       *keys,
				     uint8_t             *ivs,
				     const unsigned int  num_flows,
				     uint8_t             *checkbits = 0)
*/
__global__ void 
AES_cbc_128_encrypt_kernel_SharedMem_cbc(
					 const uint8_t		 *in_all,
				     uint8_t             *out_all,
				     size_t				 *input_size_arr, 
				     size_t				 *output_size_arr,
				     int				 num_flows, 
				     uint8_t 			 *checkbits,
					 int				 *key_idxs,
					 struct aes_sa_entry *key_array,
					 uint8_t			 *ivs,
				     const uint32_t      *pkt_offset
					 )
{
	__shared__ uint32_t shared_Te0[256];
	__shared__ uint32_t shared_Te1[256];
	__shared__ uint32_t shared_Te2[256];
	__shared__ uint32_t shared_Te3[256];
	__shared__ uint32_t shared_Rcon[10];

	/* computer the thread id */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= num_flows)
		return;
	
	/* initialize T boxes */
	for (unsigned i = 0 ; i *blockDim.x < 256 ; i++) {
		unsigned index = threadIdx.x + i * blockDim.x;
		if (index >= num_flows)
			break;
		shared_Te0[index] = Te0_ConstMem[index];
		shared_Te1[index] = Te1_ConstMem[index];
		shared_Te2[index] = Te2_ConstMem[index];
		shared_Te3[index] = Te3_ConstMem[index];
	}

	for(unsigned  i = 0;  i * blockDim.x < 10; i++){
		int index = threadIdx.x + blockDim.x * i;
		if(index < 10){
			shared_Rcon[index] = rcon[index];
		}
	}

	/* make sure T boxes have been initialized. */
	__syncthreads();

	/* Locate data */
	const uint8_t *in  = pkt_offset[idx] + in_all;
	uint8_t *out       = pkt_offset[idx] + out_all;

/*
	int temp = key_idxs[idx];
	assert(temp == key_array[temp].entry_idx);
	assert(key_array[temp].aes_key != NULL);
*/

	const uint8_t 	*key	= key_array[key_idxs[idx]].aes_key;
	uint8_t 		*ivec	= idx * AES_BLOCK_SIZE + ivs; 

	/* Encrypt using cbc mode */
	unsigned long len = pkt_offset[idx + 1] - pkt_offset[idx];
	const unsigned char *iv = ivec;

	while (len >= AES_BLOCK_SIZE) {
		*((uint64_t*)out)       = *((uint64_t*)in)       ^ *((uint64_t*)iv);
		*(((uint64_t*)out) + 1) = *(((uint64_t*)in) + 1) ^ *(((uint64_t*)iv) + 1);

		AES_128_encrypt_cbc(out, out, key,
				shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);
		iv = out;
		len -= AES_BLOCK_SIZE;
		in  += AES_BLOCK_SIZE;
		out += AES_BLOCK_SIZE;
	}

	if (len) {
		for(unsigned n = 0; n < len; ++n)
			out[n] = in[n] ^ iv[n];
		for(unsigned n = len; n < AES_BLOCK_SIZE; ++n)
			out[n] = iv[n];
		AES_128_encrypt_cbc(out, out, key,
				shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);
		iv = out;
	}

	*((uint4*)ivec) = *((uint4*)iv);
	
	__syncthreads();
	if (threadIdx.x == 0 && checkbits != 0)
		*(checkbits + blockIdx.x) = 1;
}



__global__
void AES_cbc_128_decrypt_kernel_SharedMem_cbc(const uint8_t  *in_all,
					  uint8_t        *out_all,
					  uint8_t        *keys,
					  uint8_t        *ivs,
					  uint16_t       *pkt_index,
					  unsigned long  block_count,
					  uint8_t        *checkbits = 0
					  )
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ uint32_t shared_Td0[256];
	__shared__ uint32_t shared_Td1[256];
	__shared__ uint32_t shared_Td2[256];
	__shared__ uint32_t shared_Td3[256];
	__shared__ uint8_t  shared_Td4[256];
	__shared__ uint32_t shared_Rcon[10];
	__shared__ uint32_t shared_Te0[256];
	__shared__ uint32_t shared_Te1[256];
	__shared__ uint32_t shared_Te2[256];
	__shared__ uint32_t shared_Te3[256];

	/* computer the thread id */




	/* initialize T boxes */
	for (unsigned i = 0 ; i *blockDim.x < 256 ; i++) {
		unsigned index = threadIdx.x + i * blockDim.x;
		if (index >= 256)
			break;
		shared_Te0[index] = Te0_ConstMem[index];
		shared_Te1[index] = Te1_ConstMem[index];
		shared_Te2[index] = Te2_ConstMem[index];
		shared_Te3[index] = Te3_ConstMem[index];
		shared_Td0[index] = Td0_ConstMem[index];
		shared_Td1[index] = Td1_ConstMem[index];
		shared_Td2[index] = Td2_ConstMem[index];
		shared_Td3[index] = Td3_ConstMem[index];
		shared_Td4[index] = Td4_ConstMem[index];

	}

	for(unsigned  i = 0;  i * blockDim.x < 10; i++){
		int index = threadIdx.x + blockDim.x * i;
		if(index < 10){
			shared_Rcon[index] = rcon[index];
		}
	}

	for (unsigned i = 0; i * blockDim.x < 10; i++) {
		int index = threadIdx.x + blockDim.x * i;
		if (index < 10) {
			shared_Rcon[index] = rcon[index];
		}
	}

	__syncthreads();
	if (idx >= block_count)
		return;

	/* Locate data */
	const uint8_t *in = idx * AES_BLOCK_SIZE + in_all;
	uint8_t      *out = idx * AES_BLOCK_SIZE + out_all;
	uint16_t packet_index = pkt_index[idx];

	uint32_t rk[4];
	rk[0] = *((uint32_t*)(keys + 16 * packet_index));
	rk[1] = *((uint32_t*)(keys + 16 * packet_index + 4));
	rk[2] = *((uint32_t*)(keys + 16 * packet_index + 8));
	rk[3] = *((uint32_t*)(keys + 16 * packet_index + 12));

	uint8_t *ivec = packet_index * AES_BLOCK_SIZE + ivs;

	/* Decrypt using cbc mode */
	const unsigned char *iv;
	if (idx == 0 || pkt_index[idx] != pkt_index[idx-1])
		iv = ivec;
	else
		iv = in - AES_BLOCK_SIZE;

	AES_128_decrypt_cbc(in, out, rk,
			shared_Td0, shared_Td1, shared_Td2, shared_Td3, shared_Td4,
			shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);

	*((uint64_t*)out)       = *((uint64_t*)out)       ^ *((uint64_t*)iv);
	*(((uint64_t*)out) + 1) = *(((uint64_t*)out) + 1) ^ *(((uint64_t*)iv) + 1);

	__syncthreads();
	if (threadIdx.x == 0 && checkbits != 0)
		*(checkbits + blockIdx.x) = 1;
}


/*******************************************************************
  AES ECB kernel
******************************************************************/
__global__ void
AES_ecb_encrypt_kernel_cbc(const uint8_t  *in_all,
		       uint8_t        *out_all,
		       const uint8_t  *keys,
		       uint16_t       *pkt_index,
		       unsigned long  block_count
		       )
{
	__shared__ uint32_t shared_Te0[256];
	__shared__ uint32_t shared_Te1[256];
	__shared__ uint32_t shared_Te2[256];
	__shared__ uint32_t shared_Te3[256];
	__shared__ uint32_t shared_Rcon[10];


	/* computer the thread id */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	/* initialize T boxes, #threads in block should be larger than 256 */
	for (unsigned i = 0; i * blockDim.x < 256; i++) {
		unsigned index = i * blockDim.x + threadIdx.x;
		if (index >= 256)
			break;
		shared_Te0[index] = Te0_ConstMem[index];
		shared_Te1[index] = Te1_ConstMem[index];
		shared_Te2[index] = Te2_ConstMem[index];
		shared_Te3[index] = Te3_ConstMem[index];

	}

	for (unsigned i = 0; i * blockDim.x < 10; i++) {
		unsigned index = threadIdx.x + blockDim.x * i;
		if (index < 10) {
			shared_Rcon[index] = rcon[index];
		}
	}

	if (idx >= block_count)
		return;

	/* make sure T boxes have been initialized. */
	__syncthreads();

	/* Locate data */
	const uint8_t  *in = idx * AES_BLOCK_SIZE + in_all;
	uint8_t       *out = idx * AES_BLOCK_SIZE + out_all;
	uint16_t  pktIndex = pkt_index[idx];
	const uint8_t *key = pktIndex * 16 + keys;

	AES_128_encrypt_cbc(in, out, key,
			shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);
}

/**************************************************************************
 Exported C++ function wrapper function for CUDA kernel
***************************************************************************/

/* 
 * Sangwook: Those wrapper functions are not used in NBA.
void AES_cbc_128_decrypt_gpu(const uint8_t        *in_d,
			     uint8_t              *out_d,
			     uint8_t              *keys_d,
			     uint8_t              *ivs_d,
			     uint16_t             *pkt_index_d,
			     unsigned long        block_count,
			     uint8_t              *checkbits_d,
			     const unsigned int   threads_per_blk,
			     cudaStream_t         stream )
{
	unsigned int num_cuda_blks = (block_count+threads_per_blk - 1) / threads_per_blk;
	if (stream == 0) {
		AES_cbc_128_decrypt_kernel_SharedMem_cbc<<<num_cuda_blks, threads_per_blk>>>(
		    in_d, out_d, keys_d, ivs_d, pkt_index_d, block_count, checkbits_d);
	} else {
		AES_cbc_128_decrypt_kernel_SharedMem_cbc<<<num_cuda_blks, threads_per_blk, 0, stream>>>(
		    in_d, out_d, keys_d, ivs_d, pkt_index_d, block_count, checkbits_d);
	}
}


void AES_cbc_128_encrypt_gpu(const uint8_t      *in_d,
			     uint8_t            *out_d,
			     const uint32_t     *pkt_offset_d,
			     const uint8_t      *keys_d,
			     uint8_t            *ivs_d,
			     const              unsigned int num_flows,
			     uint8_t            *checkbits_d,
			     const unsigned int threads_per_blk,
			     cudaStream_t stream)
{
	unsigned int num_cuda_blks = (num_flows+threads_per_blk - 1) / threads_per_blk;
	if (stream == 0) {
		AES_cbc_128_encrypt_kernel_SharedMem_cbc<<<num_cuda_blks, threads_per_blk>>>(
		    in_d, out_d, pkt_offset_d, keys_d, ivs_d, num_flows, checkbits_d);
	} else {
		AES_cbc_128_encrypt_kernel_SharedMem_cbc<<<num_cuda_blks, threads_per_blk, 0, stream>>>(
		    in_d, out_d, pkt_offset_d, keys_d, ivs_d, num_flows, checkbits_d);
	}
}

void AES_ecb_128_encrypt_gpu(const uint8_t      *in_d,
			     uint8_t            *out_d,
			     const uint8_t      *keys_d,
			     uint16_t           *pkt_index_d,
			     unsigned long      block_count,
			     const unsigned int threads_per_blk,
			     cudaStream_t stream)
{
	unsigned int num_cuda_blks = (block_count + threads_per_blk - 1) / threads_per_blk;
	if (stream == 0) {
		AES_ecb_encrypt_kernel_cbc<<<num_cuda_blks, threads_per_blk>>>(
		    in_d, out_d, keys_d, pkt_index_d, block_count);
	} else {
		AES_ecb_encrypt_kernel_cbc<<<num_cuda_blks, threads_per_blk, 0, stream>>>(
		    in_d, out_d, keys_d, pkt_index_d, block_count);
	}
}
*/

/**************************************************************************
Key Setup for Decryption
***************************************************************************/
void AES_decrypt_key_prepare_cbc(uint8_t        *dec_key,
			     const uint8_t  *enc_key,
			     unsigned int   key_bits)
{
	uint32_t rk_buf[60];
	uint32_t *rk = rk_buf;
	int i = 0;
	uint32_t temp;

	rk[0] = GETU32_cbc(enc_key     );
	rk[1] = GETU32_cbc(enc_key +  4);
	rk[2] = GETU32_cbc(enc_key +  8);
	rk[3] = GETU32_cbc(enc_key + 12);
	if (key_bits == 128) {
		for (;;) {
			temp  = rk[3];
			rk[4] = rk[0] ^
				(Te4[(temp >> 16) & 0xff] & 0xff000000) ^
				(Te4[(temp >>  8) & 0xff] & 0x00ff0000) ^
				(Te4[(temp      ) & 0xff] & 0x0000ff00) ^
				(Te4[(temp >> 24)       ] & 0x000000ff) ^
				rcon_host[i];
			rk[5] = rk[1] ^ rk[4];
			rk[6] = rk[2] ^ rk[5];
			rk[7] = rk[3] ^ rk[6];
			if (++i == 10) {
				rk += 4;
				goto end;
			}
			rk += 4;
		}
	}
	rk[4] = GETU32_cbc(enc_key + 16);
	rk[5] = GETU32_cbc(enc_key + 20);
	if (key_bits == 192) {
		for (;;) {
			temp = rk[ 5];
			rk[ 6] = rk[ 0] ^
				(Te4[(temp >> 16) & 0xff] & 0xff000000) ^
				(Te4[(temp >>  8) & 0xff] & 0x00ff0000) ^
				(Te4[(temp      ) & 0xff] & 0x0000ff00) ^
				(Te4[(temp >> 24)       ] & 0x000000ff) ^
				rcon_host[i];
			rk[ 7] = rk[ 1] ^ rk[ 6];
			rk[ 8] = rk[ 2] ^ rk[ 7];
			rk[ 9] = rk[ 3] ^ rk[ 8];
			if (++i == 8) {
				rk += 6;
				goto end;

			}
			rk[10] = rk[ 4] ^ rk[ 9];
			rk[11] = rk[ 5] ^ rk[10];
			rk += 6;
		}
	}
	rk[6] = GETU32_cbc(enc_key + 24);
	rk[7] = GETU32_cbc(enc_key + 28);
	if (key_bits == 256) {
		for (;;) {
			temp = rk[ 7];
			rk[ 8] = rk[ 0] ^
				(Te4[(temp >> 16) & 0xff] & 0xff000000) ^
				(Te4[(temp >>  8) & 0xff] & 0x00ff0000) ^
				(Te4[(temp      ) & 0xff] & 0x0000ff00) ^
				(Te4[(temp >> 24)       ] & 0x000000ff) ^
				rcon_host[i];
			rk[ 9] = rk[ 1] ^ rk[ 8];
			rk[10] = rk[ 2] ^ rk[ 9];
			rk[11] = rk[ 3] ^ rk[10];
			if (++i == 7) {
				rk += 8;
				goto end;
			}
			temp = rk[11];
			rk[12] = rk[ 4] ^
				(Te4[(temp >> 24)       ] & 0xff000000) ^
				(Te4[(temp >> 16) & 0xff] & 0x00ff0000) ^
				(Te4[(temp >>  8) & 0xff] & 0x0000ff00) ^
				(Te4[(temp      ) & 0xff] & 0x000000ff);
			rk[13] = rk[ 5] ^ rk[12];
			rk[14] = rk[ 6] ^ rk[13];
			rk[15] = rk[ 7] ^ rk[14];

			rk += 8;
		}
	}
 end:
	memcpy(dec_key, rk, 16);
}

/**************************************************************************
 Experimental Codes
***************************************************************************/

__global__ void computeAES_CBC(
		uint8_t* input_buf, uint8_t *output_buf,
        size_t *input_size_arr, size_t *output_size_arr,
        int N, uint8_t *checkbits_d,
        const uint8_t* __restrict__ ivs,
        const int32_t* __restrict__ key_idxs, const struct aes_sa_entry* __restrict__ aes_key_array,
        const int32_t* __restrict__ offsets)
{
	/* computer the thread id */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int index = idx;

    if (idx < N) {
        /* Locate data */
        const uint8_t *in  = input_buf + offsets[idx];
        uint8_t *out       = output_buf + offsets[idx];

        __shared__ uint32_t shared_Te0[256];
        __shared__ uint32_t shared_Te1[256];
        __shared__ uint32_t shared_Te2[256];
        __shared__ uint32_t shared_Te3[256];
        __shared__ uint32_t shared_Rcon[10];

    /*
        int temp = key_idxs[idx];
        assert(temp == key_array[temp].entry_idx);
        assert(key_array[temp].aes_key != NULL);
    */

        const uint8_t   *key = (const uint8_t*) aes_key_array[key_idxs[index]].aes_key;
        uint8_t 		*ivec	= (uint8_t*) (idx * AES_BLOCK_SIZE + ivs); 

        /* Encrypt using cbc mode */
        unsigned long len = (unsigned long) input_size_arr[index];
        const unsigned char *iv = ivec;

        while (len >= AES_BLOCK_SIZE) {
            *((uint64_t*)out)       = *((uint64_t*)in)       ^ *((uint64_t*)iv);
            *(((uint64_t*)out) + 1) = *(((uint64_t*)in) + 1) ^ *(((uint64_t*)iv) + 1);

            AES_128_encrypt_cbc(out, out, key,
                    shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);
            iv = out;
            len -= AES_BLOCK_SIZE;
            in  += AES_BLOCK_SIZE;
            out += AES_BLOCK_SIZE;
        }
        
        if (len) {
            for(unsigned n = 0; n < len; ++n)
                out[n] = in[n] ^ iv[n];
            for(unsigned n = len; n < AES_BLOCK_SIZE; ++n)
                out[n] = iv[n];
            AES_128_encrypt_cbc(out, out, key,
                    shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);
            iv = out;
        }

        *((uint4*)ivec) = *((uint4*)iv);
    }

	__syncthreads();
	if (threadIdx.x == 0 && checkbits_d != 0)
		*(checkbits_d + blockIdx.x) = 1;
}


/* Among AES_cbc_128_decryption, AES_cbc_128_encryption,
 * AES_ecb_128_encryption and AES_decrypt_key_prepare_cbc(),
 * AES_cbc_128_encrypt_gpu() is only used in NBA, for now. */
 
void *nshader::ipsec_aes_encryption_cbc_get_cuda_kernel() {
	return reinterpret_cast<void *> (computeAES_CBC);
}

