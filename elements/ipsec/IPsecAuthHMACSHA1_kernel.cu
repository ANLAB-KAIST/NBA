#include <cstdint>
#include <cassert>
#include <cuda.h>
#include <nba/engines/cuda/utils.hh>
#include <nba/core/errors.hh>
#include <nba/core/accumidx.hh>
#include <nba/framework/datablock_shared.hh>

#include "IPsecAuthHMACSHA1_kernel.hh"

/* The index is given by the order in get_used_datablocks(). */
#define dbid_enc_payloads_d (0)
#define dbid_flow_ids_d     (1)

#define SHA1_THREADS_PER_BLK 32

extern "C" {

//__global__ uint32_t d_pad_buffer[16 * 2 * MAX_CHUNK_SIZE * MAX_GROUP_SIZE];

__device__ static uint32_t swap(uint32_t v) {
    return ((v & 0x000000ffU) << 24) | ((v & 0x0000ff00U) << 8)
            | ((v & 0x00ff0000U) >> 8) | ((v & 0xff000000U) >> 24);
}

typedef struct hash_digest {
    uint32_t h1;
    uint32_t h2;
    uint32_t h3;
    uint32_t h4;
    uint32_t h5;
} hash_digest_t;

#define HMAC

__inline__ __device__ static void getBlock(char* buf, int offset, int len, uint32_t* dest)
{
    uint32_t *tmp;
    unsigned int tempbuf[16];

    tmp = (uint32_t*) (buf + offset);
    //printf("%d %d\n", offset, len);
    if (offset + 64 <= len) {
        //printf("--0--\n");
#pragma unroll 16
        for (int i = 0; i < 16; i++) {
            dest[i] = swap(tmp[i]);
        }
    } else if (len > offset && (len - offset) < 56) { //case 1 enough space in last block for padding
        //prtinf("--1--\n");
        int i;
        for (i = 0; i < (len - offset) / 4; i++) {

            //printf("%d %d\n",offset,i);
            //printf("%p %p\n", buf, dest);

            //tempbuf[i] = buf[i];
            tempbuf[i] = swap(tmp[i]);
        }
        //printf("len%%4 %d\n",len%4);
        switch (len % 4) {
        case 0:
            tempbuf[i] = swap(0x00000080);
            i++;
            break;
        case 1:
            tempbuf[i] = swap(0x00008000 | (tmp[i] & 0x000000FF));
            i++;
            break;
        case 2:
            tempbuf[i] = swap(0x00800000 | (tmp[i] & 0x0000FFFF));
            i++;
            break;
        case 3:
            tempbuf[i] = swap(0x80000000 | (tmp[i] & 0x00FFFFFF));
            i++;
            break;
        };
        for (; i < 14; i++) {
            tempbuf[i] = 0;
        }
#pragma unroll 14
        for (i = 0; i < 14; i++) {
            dest[i] = tempbuf[i];
        }
        dest[14] = 0x00000000;
#ifndef HMAC
        dest[15] = len * 8;
#else
        dest[15] = (len + 64) * 8;
#endif

    } else if (len > offset && (len - offset) >= 56) { //case 2 not enough space in last block (containing message) for padding
        //printf("--2--\n");
        int i;
        for (i = 0; i < (len - offset) / 4; i++) {
            tempbuf[i] = swap(tmp[i]);
        }
        switch (len % 4) {
        case 0:
            tempbuf[i] = swap(0x00000080);
            i++;
            break;
        case 1:
            tempbuf[i] = swap(0x00008000 | (tmp[i] & 0x000000FF));
            i++;
            break;
        case 2:
            tempbuf[i] = swap(0x00800000 | (tmp[i] & 0x0000FFFF));
            i++;
            break;
        case 3:
            tempbuf[i] = swap(0x80000000 | (tmp[i] & 0x00FFFFFF));
            i++;
            break;
        };

        for (; i < 16; i++) {
            tempbuf[i] = 0x00000000;
        }

#pragma unroll 16
        for (i = 0; i < 16; i++) {
            dest[i] = tempbuf[i];
        }

    } else if (offset == len) { //message end is aligned in 64 bytes
        //printf("--3--\n");
        dest[0] = swap(0x00000080);
#pragma unroll 13
        for (int i = 1; i < 14; i++)
            dest[i] = 0x00000000;
        dest[14] = 0x00000000;
#ifndef HMAC
        dest[15] = len * 8;
#else
        dest[15] = (len + 64) * 8;
#endif

    } else if (offset > len) { //the last block in case 2
        //printf("--4--\n");
#pragma unroll 14
        for (int i = 0; i < 14; i++)
            dest[i] = 0x00000000;
        dest[14] = 0x00000000;
#ifndef HMAC
        dest[15] = len * 8;
#else
        dest[15] = (len + 64) * 8;
#endif

    } else {
        printf("Not supposed to happen\n");
    }
}

__device__ static void computeSHA1Block(char* in, uint32_t* w, int offset, int len,
        hash_digest_t &h) {
    uint32_t a = h.h1;
    uint32_t b = h.h2;
    uint32_t c = h.h3;
    uint32_t d = h.h4;
    uint32_t e = h.h5;
    uint32_t f;
    uint32_t k;
    uint32_t temp;

    getBlock(in, offset, len, w);

    //for (int i = 0; i < 16 ; i++) {
    //  printf("%0X\n", w[i]);
    //}
    //printf("\n");

    k = 0x5A827999;
    //0 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[0];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[0] = w[13] ^ w[8] ^ w[2] ^ w[0];
    w[0] = w[0] << 1 | w[0] >> 31;

    //1 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[1];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[1] = w[14] ^ w[9] ^ w[3] ^ w[1];
    w[1] = w[1] << 1 | w[1] >> 31;

    //2 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[2];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[2] = w[15] ^ w[10] ^ w[4] ^ w[2];
    w[2] = w[2] << 1 | w[2] >> 31;

    //3 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[3];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[3] = w[0] ^ w[11] ^ w[5] ^ w[3];
    w[3] = w[3] << 1 | w[3] >> 31;

    //4 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[4];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[4] = w[1] ^ w[12] ^ w[6] ^ w[4];
    w[4] = w[4] << 1 | w[4] >> 31;

    //5 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[5];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[5] = w[2] ^ w[13] ^ w[7] ^ w[5];
    w[5] = w[5] << 1 | w[5] >> 31;

    //6 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[6];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[6] = w[3] ^ w[14] ^ w[8] ^ w[6];
    w[6] = w[6] << 1 | w[6] >> 31;

    //7 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[7];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[7] = w[4] ^ w[15] ^ w[9] ^ w[7];
    w[7] = w[7] << 1 | w[7] >> 31;

    //8 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[8];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[8] = w[5] ^ w[0] ^ w[10] ^ w[8];
    w[8] = w[8] << 1 | w[8] >> 31;

    //9 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[9];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[9] = w[6] ^ w[1] ^ w[11] ^ w[9];
    w[9] = w[9] << 1 | w[9] >> 31;

    //10 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[10];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[10] = w[7] ^ w[2] ^ w[12] ^ w[10];
    w[10] = w[10] << 1 | w[10] >> 31;

    //11 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[11];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[11] = w[8] ^ w[3] ^ w[13] ^ w[11];
    w[11] = w[11] << 1 | w[11] >> 31;

    //12 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[12];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[12] = w[9] ^ w[4] ^ w[14] ^ w[12];
    w[12] = w[12] << 1 | w[12] >> 31;

    //13 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[13];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[13] = w[10] ^ w[5] ^ w[15] ^ w[13];
    w[13] = w[13] << 1 | w[13] >> 31;

    //14 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[14];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[14] = w[11] ^ w[6] ^ w[0] ^ w[14];
    w[14] = w[14] << 1 | w[14] >> 31;

    //15 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[15];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[15] = w[12] ^ w[7] ^ w[1] ^ w[15];
    w[15] = w[15] << 1 | w[15] >> 31;

    //16 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[0];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[0] = w[13] ^ w[8] ^ w[2] ^ w[0];
    w[0] = w[0] << 1 | w[0] >> 31;

    //17 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[1];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[1] = w[14] ^ w[9] ^ w[3] ^ w[1];
    w[1] = w[1] << 1 | w[1] >> 31;

    //18 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[2];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[2] = w[15] ^ w[10] ^ w[4] ^ w[2];
    w[2] = w[2] << 1 | w[2] >> 31;

    //19 of 0-20
    f = (b & c) | ((~b) & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[3];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[3] = w[0] ^ w[11] ^ w[5] ^ w[3];
    w[3] = w[3] << 1 | w[3] >> 31;

    k = 0x6ED9EBA1;
    //20 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[4];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[4] = w[1] ^ w[12] ^ w[6] ^ w[4];
    w[4] = w[4] << 1 | w[4] >> 31;

    //21 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[5];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[5] = w[2] ^ w[13] ^ w[7] ^ w[5];
    w[5] = w[5] << 1 | w[5] >> 31;

    //22 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[6];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[6] = w[3] ^ w[14] ^ w[8] ^ w[6];
    w[6] = w[6] << 1 | w[6] >> 31;

    //23 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[7];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[7] = w[4] ^ w[15] ^ w[9] ^ w[7];
    w[7] = w[7] << 1 | w[7] >> 31;

    //24 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[8];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[8] = w[5] ^ w[0] ^ w[10] ^ w[8];
    w[8] = w[8] << 1 | w[8] >> 31;

    //25 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[9];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[9] = w[6] ^ w[1] ^ w[11] ^ w[9];
    w[9] = w[9] << 1 | w[9] >> 31;

    //26 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[10];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[10] = w[7] ^ w[2] ^ w[12] ^ w[10];
    w[10] = w[10] << 1 | w[10] >> 31;

    //27 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[11];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[11] = w[8] ^ w[3] ^ w[13] ^ w[11];
    w[11] = w[11] << 1 | w[11] >> 31;

    //28 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[12];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[12] = w[9] ^ w[4] ^ w[14] ^ w[12];
    w[12] = w[12] << 1 | w[12] >> 31;

    //29 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[13];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[13] = w[10] ^ w[5] ^ w[15] ^ w[13];
    w[13] = w[13] << 1 | w[13] >> 31;

    //30 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[14];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[14] = w[11] ^ w[6] ^ w[0] ^ w[14];
    w[14] = w[14] << 1 | w[14] >> 31;

    //31 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[15];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[15] = w[12] ^ w[7] ^ w[1] ^ w[15];
    w[15] = w[15] << 1 | w[15] >> 31;

    //32 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[0];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[0] = w[13] ^ w[8] ^ w[2] ^ w[0];
    w[0] = w[0] << 1 | w[0] >> 31;

    //33 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[1];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[1] = w[14] ^ w[9] ^ w[3] ^ w[1];
    w[1] = w[1] << 1 | w[1] >> 31;

    //34 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[2];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[2] = w[15] ^ w[10] ^ w[4] ^ w[2];
    w[2] = w[2] << 1 | w[2] >> 31;

    //35 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[3];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[3] = w[0] ^ w[11] ^ w[5] ^ w[3];
    w[3] = w[3] << 1 | w[3] >> 31;

    //36 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[4];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[4] = w[1] ^ w[12] ^ w[6] ^ w[4];
    w[4] = w[4] << 1 | w[4] >> 31;

    //37 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[5];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[5] = w[2] ^ w[13] ^ w[7] ^ w[5];
    w[5] = w[5] << 1 | w[5] >> 31;

    //38 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[6];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[6] = w[3] ^ w[14] ^ w[8] ^ w[6];
    w[6] = w[6] << 1 | w[6] >> 31;

    //39 of 20-40
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[7];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[7] = w[4] ^ w[15] ^ w[9] ^ w[7];
    w[7] = w[7] << 1 | w[7] >> 31;

    k = 0x8F1BBCDC;
    //40 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[8];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[8] = w[5] ^ w[0] ^ w[10] ^ w[8];
    w[8] = w[8] << 1 | w[8] >> 31;

    //41 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[9];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[9] = w[6] ^ w[1] ^ w[11] ^ w[9];
    w[9] = w[9] << 1 | w[9] >> 31;

    //42 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[10];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[10] = w[7] ^ w[2] ^ w[12] ^ w[10];
    w[10] = w[10] << 1 | w[10] >> 31;

    //43 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[11];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[11] = w[8] ^ w[3] ^ w[13] ^ w[11];
    w[11] = w[11] << 1 | w[11] >> 31;

    //44 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[12];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[12] = w[9] ^ w[4] ^ w[14] ^ w[12];
    w[12] = w[12] << 1 | w[12] >> 31;

    //45 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[13];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[13] = w[10] ^ w[5] ^ w[15] ^ w[13];
    w[13] = w[13] << 1 | w[13] >> 31;

    //46 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[14];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[14] = w[11] ^ w[6] ^ w[0] ^ w[14];
    w[14] = w[14] << 1 | w[14] >> 31;

    //47 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[15];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[15] = w[12] ^ w[7] ^ w[1] ^ w[15];
    w[15] = w[15] << 1 | w[15] >> 31;

    //48 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[0];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[0] = w[13] ^ w[8] ^ w[2] ^ w[0];
    w[0] = w[0] << 1 | w[0] >> 31;

    //49 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[1];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[1] = w[14] ^ w[9] ^ w[3] ^ w[1];
    w[1] = w[1] << 1 | w[1] >> 31;

    //50 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[2];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[2] = w[15] ^ w[10] ^ w[4] ^ w[2];
    w[2] = w[2] << 1 | w[2] >> 31;

    //51 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[3];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[3] = w[0] ^ w[11] ^ w[5] ^ w[3];
    w[3] = w[3] << 1 | w[3] >> 31;

    //52 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[4];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[4] = w[1] ^ w[12] ^ w[6] ^ w[4];
    w[4] = w[4] << 1 | w[4] >> 31;

    //53 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[5];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[5] = w[2] ^ w[13] ^ w[7] ^ w[5];
    w[5] = w[5] << 1 | w[5] >> 31;

    //54 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[6];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[6] = w[3] ^ w[14] ^ w[8] ^ w[6];
    w[6] = w[6] << 1 | w[6] >> 31;

    //55 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[7];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[7] = w[4] ^ w[15] ^ w[9] ^ w[7];
    w[7] = w[7] << 1 | w[7] >> 31;

    //56 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[8];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[8] = w[5] ^ w[0] ^ w[10] ^ w[8];
    w[8] = w[8] << 1 | w[8] >> 31;

    //57 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[9];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[9] = w[6] ^ w[1] ^ w[11] ^ w[9];
    w[9] = w[9] << 1 | w[9] >> 31;

    //58 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[10];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[10] = w[7] ^ w[2] ^ w[12] ^ w[10];
    w[10] = w[10] << 1 | w[10] >> 31;

    //59 of 40-60
    f = (b & c) | (b & d) | (c & d);
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[11];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[11] = w[8] ^ w[3] ^ w[13] ^ w[11];
    w[11] = w[11] << 1 | w[11] >> 31;

    k = 0xCA62C1D6;

    //60 of 60-64
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[12];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[12] = w[9] ^ w[4] ^ w[14] ^ w[12];
    w[12] = w[12] << 1 | w[12] >> 31;

    //61 of 60-64
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[13];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[13] = w[10] ^ w[5] ^ w[15] ^ w[13];
    w[13] = w[13] << 1 | w[13] >> 31;

    //62 of 60-64
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[14];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[14] = w[11] ^ w[6] ^ w[0] ^ w[14];
    w[14] = w[14] << 1 | w[14] >> 31;

    //63 of 60-64
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[15];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    w[15] = w[12] ^ w[7] ^ w[1] ^ w[15];
    w[15] = w[15] << 1 | w[15] >> 31;

    //64 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[0];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //65 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[1];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //66 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[2];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //67 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[3];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //68 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[4];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //69 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[5];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //70 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[6];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //71 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[7];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //72 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[8];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //73 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[9];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //74 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[10];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //75 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[11];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //76 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[12];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //77 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[13];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //78 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[14];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    //79 of 64-80
    f = b ^ c ^ d;
    temp = ((a << 5) | (a >> 27)) + f + e + k + w[15];
    e = d;
    d = c;
    c = (b << 30) | (b >> 2);
    b = a;
    a = temp;

    h.h1 += a;
    h.h2 += b;
    h.h3 += c;
    h.h4 += d;
    h.h5 += e;

}
/*
 __global__ void computeSHA1(char* buf, int *offsets, int *len, char* output, int N)
 {

 //__shared__ uint32_t w_shared[16*SHA1_THREADS_PER_BLK];
 uint32_t w_register[16];

 int index = blockIdx.x * blockDim.x + threadIdx.x;
 if (index < N) {
 uint32_t *w = w_register;//w_shared + 16*threadIdx.x;
 hash_digest_t h;
 h.h1 = 0x67452301;
 h.h2 = 0xEFCDAB89;
 h.h3 = 0x98BADCFE;
 h.h4 = 0x10325476;
 h.h5 = 0xC3D2E1F0;

 int num_iter = (len[index]+63+9)/64;
 printf("num_iter %d\n", num_iter);
 for(int i = 0; i < num_iter; i++)
 computeSHA1Block(buf + offsets[index], w, i*64 , len[index], h);

 h.h1 = swap(h.h1);
 h.h2 = swap(h.h2);
 h.h3 = swap(h.h3);
 h.h4 = swap(h.h4);
 h.h5 = swap(h.h5);

 uint32_t * out = (uint32_t*)(output + index*20);
 *(out++) = h.h1;
 *(out++) = h.h2;
 *(out++) = h.h3;
 *(out++) = h.h4;
 *(out++) = h.h5;
 }
 }*/
/*
 some how *pad = *pad++ ^ *key++
 was optimized and does not work correctly in GPU oTL.
 */
__device__ static void xorpads(uint32_t *pad, const uint32_t* key) {
#pragma unroll 16
    for (int i = 0; i < 16; i++)
        *(pad + i) = *(pad + i) ^ *(key + i);
}

uint32_t opad[16] =
        { 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c,
          0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c,
          0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c,
          0x5c5c5c5c, };
uint32_t ipad[16] =
        { 0x36363636, 0x36363636, 0x36363636, 0x36363636, 0x36363636,
          0x36363636, 0x36363636, 0x36363636, 0x36363636, 0x36363636,
          0x36363636, 0x36363636, 0x36363636, 0x36363636, 0x36363636,
          0x36363636, };

// in: start pointer of the data to be authenticated by hsha1.
// out: start pointer of the data where hsha1 signature will be recorded.
// length: length of the data to be authenticated by hsha1.
// key: hmac key.
__device__ static void HMAC_SHA1(uint32_t *in, uint32_t *out, uint32_t length,
        const char *key) {
    uint32_t w_register[16];

    uint32_t *w = w_register; //w_shared + 16*threadIdx.x;
    hash_digest_t h;

    for (int i = 0; i < 16; i++)
        w[i] = 0x36363636;
    xorpads(w, (uint32_t*) (key));

    h.h1 = 0x67452301;
    h.h2 = 0xEFCDAB89;
    h.h3 = 0x98BADCFE;
    h.h4 = 0x10325476;
    h.h5 = 0xC3D2E1F0;

    //SHA1 compute on ipad
    computeSHA1Block((char*) w, w, 0, 64, h);

    //SHA1 compute on mesage
    int num_iter = (length + 63 + 9) / 64;
    for (int i = 0; i < num_iter; i++)
        computeSHA1Block((char*) in, w, i * 64, length, h);

    *(out) = swap(h.h1);
    *(out + 1) = swap(h.h2);
    *(out + 2) = swap(h.h3);
    *(out + 3) = swap(h.h4);
    *(out + 4) = swap(h.h5);

    h.h1 = 0x67452301;
    h.h2 = 0xEFCDAB89;
    h.h3 = 0x98BADCFE;
    h.h4 = 0x10325476;
    h.h5 = 0xC3D2E1F0;

    for (int i = 0; i < 16; i++)
        w[i] = 0x5c5c5c5c;

    xorpads(w, (uint32_t*) (key));

    //SHA 1 compute on opads
    computeSHA1Block((char*) w, w, 0, 64, h);

    //SHA 1 compute on (hash of ipad|m)
    computeSHA1Block((char*) out, w, 0, 20, h);

    *(out) = swap(h.h1);
    *(out + 1) = swap(h.h2);
    *(out + 2) = swap(h.h3);
    *(out + 3) = swap(h.h4);
    *(out + 4) = swap(h.h5);
}
#if 0
__global__ void computeHMAC_SHA1(char* buf, char* keys, uint32_t *offsets,
                                 uint32_t *lengths, uint32_t *outputs, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        printf("index%d threadid%d\n", index, threadIdx.x);
        uint32_t offset = offsets[index];
        uint32_t length = lengths[index];
        uint32_t *out = (uint32_t*) (buf + outputs[index]);
        HMAC_SHA1((uint32_t*) (buf + offset), out, length, keys + 64 * index);
    }
}

__global__ void computeHMAC_SHA1_2(char* buf, char* keys, uint32_t *offsets,
                                   uint16_t *lengths, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        printf("index%d threadid%d\n", index, threadIdx.x);
        uint32_t offset = offsets[index];
        uint32_t length = lengths[index];
        uint32_t *out = (uint32_t*) (buf + offset + length);
        HMAC_SHA1((uint32_t*) (buf + offset), out, length, keys + 64 * index);
    }
}
#endif

__global__ void computeHMAC_SHA1_3(
        struct datablock_kernel_arg **datablocks,
        uint32_t count, uint32_t *item_counts, uint32_t num_batches,
        uint8_t *checkbits_d,
        struct hmac_sa_entry *hmac_key_array)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && count != 0) {
        uint32_t batch_idx, item_idx;
        nba::error_t err;
        err = nba::get_accum_idx(item_counts, num_batches, idx, batch_idx, item_idx);
        assert(err == nba::NBA_SUCCESS);

        const struct datablock_kernel_arg *db_enc_payloads = datablocks[dbid_enc_payloads_d];
        const struct datablock_kernel_arg *db_flow_ids     = datablocks[dbid_flow_ids_d];

        const uint8_t *enc_payload_base = (uint8_t *) db_enc_payloads->batches[batch_idx].buffer_bases;
        const uintptr_t offset = (uintptr_t) db_enc_payloads->batches[batch_idx].item_offsets[item_idx].as_value<uintptr_t>();
        const uintptr_t length = (uintptr_t) db_enc_payloads->batches[batch_idx].item_sizes[item_idx];
        if (enc_payload_base != NULL && length != 0) {
            const uint64_t flow_id = ((uint64_t *) db_flow_ids->batches[batch_idx].buffer_bases)[item_idx];
            if (flow_id != 65536) {
                assert(flow_id < 1024);
                const char *hmac_key = (char *) hmac_key_array[flow_id].hmac_key;
                HMAC_SHA1((uint32_t *) (enc_payload_base + offset),
                          (uint32_t *) (enc_payload_base + offset + length),
                          length, hmac_key);
            }
        }

        __syncthreads();
        if (threadIdx.x == 0 && checkbits_d != NULL)
            checkbits_d[blockIdx.x] = 1;
    } // endif(valid-idx)
}

}

void *nba::ipsec_hsha1_encryption_get_cuda_kernel() {
    return reinterpret_cast<void *> (computeHMAC_SHA1_3);
}

// vim: ts=8 sts=4 sw=4 et tw=150
