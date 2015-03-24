#include "IPsecHMACSHA1AES_kernel_core.hh"
#include "IPsecHMACSHA1AES_kernel.hh"

#include <cuda.h>
#include "../../engines/cuda/utils.hh"

#include <stdint.h>

#include <assert.h>
#include <stdio.h>


/*******************************************************************
  HMAC-SHA1 kernel
******************************************************************/

#ifdef __DEVICE_EMULATION__
#define debugprint printf
#define EMUSYNC __syncthreads()
#else
__device__ void _NOOPfunction(char *format) {
}
__device__ void _NOOPfunction(char *format, unsigned int onearg) {
}
__device__ void _NOOPfunction(char *format, unsigned int onearg,
        unsigned int twoargs) {
}
__device__ void _NOOPfunction(char *format, char *onearg) {
}
#define EMUSYNC do {} while (0)
#define debugprint _NOOPfunction
#endif

#define SHA1_THREADS_PER_BLK 32


//__global__ uint32_t d_pad_buffer[16 * 2 * MAX_CHUNK_SIZE * MAX_GROUP_SIZE];

__device__ uint32_t swap(uint32_t v) {
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

__inline__ __device__ void getBlock(char* buf, int offset, int len,
        uint32_t* dest) {
    uint32_t *tmp;

    unsigned int tempbuf[16];

    tmp = (uint32_t*) (buf + offset);
    debugprint("%d %d\n", offset, len);
    if (offset + 64 <= len) {
        debugprint("--0--\n");
#pragma unroll 16
        for (int i = 0; i < 16; i++) {
            dest[i] = swap(tmp[i]);
        }
    } else if (len > offset && (len - offset) < 56) { //case 1 enough space in last block for padding
        debugprint("--1--\n");
        int i;
        for (i = 0; i < (len - offset) / 4; i++) {

            //debugprint("%d %d\n",offset,i);
            //debugprint("%p %p\n", buf, dest);

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
        debugprint("--2--\n");
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
        debugprint("--3--\n");
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
        debugprint("--4--\n");
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
        debugprint("Not supposed to happen\n");
    }
}

__device__ void computeSHA1Block(char* in, uint32_t* w, int offset, int len,
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
    //  debugprint("%0X\n", w[i]);
    //}
    //debugprint("\n");

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
 debugprint("num_iter %d\n", num_iter);
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
__device__ void xorpads(uint32_t *pad, uint32_t* key) {
#pragma unroll 16
    for (int i = 0; i < 16; i++)
        *(pad + i) = *(pad + i) ^ *(key + i);
}
/*
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
*/
// in: start pointer of the data to be authenticated by hsha1.
// out: start pointer of the data where hsha1 signature will be recorded.
// length: length of the data to be authenticated by hsha1.
// key: hmac key.
__device__ void HMAC_SHA1(uint32_t *in, uint32_t *out, uint32_t length,
        char *key) {
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








/*******************************************************************
  AES CBC kernel
******************************************************************/

/* former prototype
__global__ void
AES_cbc_128_encrypt_kernel_SharedMem(const uint8_t       *in_all,
				     uint8_t             *out_all,
				     const uint32_t      *pkt_offset,
				     const uint8_t       *keys,
				     uint8_t             *ivs,
				     const unsigned int  num_flows,
				     uint8_t             *checkbits = 0)
*/
__global__ void 
AES_cbc_128_encrypt_kernel_SharedMem(
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

		AES_128_encrypt(out, out, key,
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
		AES_128_encrypt(out, out, key,
				shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);
		iv = out;
	}

	*((uint4*)ivec) = *((uint4*)iv);
	
	__syncthreads();
	if (threadIdx.x == 0 && checkbits != 0)
		*(checkbits + blockIdx.x) = 1;
}



__global__
void AES_cbc_128_decrypt_kernel_SharedMem(const uint8_t  *in_all,
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

	AES_128_decrypt(in, out, rk,
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
AES_ecb_encrypt_kernel(const uint8_t  *in_all,
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

	AES_128_encrypt(in, out, key,
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
		AES_cbc_128_decrypt_kernel_SharedMem<<<num_cuda_blks, threads_per_blk>>>(
		    in_d, out_d, keys_d, ivs_d, pkt_index_d, block_count, checkbits_d);
	} else {
		AES_cbc_128_decrypt_kernel_SharedMem<<<num_cuda_blks, threads_per_blk, 0, stream>>>(
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
		AES_cbc_128_encrypt_kernel_SharedMem<<<num_cuda_blks, threads_per_blk>>>(
		    in_d, out_d, pkt_offset_d, keys_d, ivs_d, num_flows, checkbits_d);
	} else {
		AES_cbc_128_encrypt_kernel_SharedMem<<<num_cuda_blks, threads_per_blk, 0, stream>>>(
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
		AES_ecb_encrypt_kernel<<<num_cuda_blks, threads_per_blk>>>(
		    in_d, out_d, keys_d, pkt_index_d, block_count);
	} else {
		AES_ecb_encrypt_kernel<<<num_cuda_blks, threads_per_blk, 0, stream>>>(
		    in_d, out_d, keys_d, pkt_index_d, block_count);
	}
}
*/

/**************************************************************************
Key Setup for Decryption
***************************************************************************/
void AES_decrypt_key_prepare(uint8_t        *dec_key,
			     const uint8_t  *enc_key,
			     unsigned int   key_bits)
{
	uint32_t rk_buf[60];
	uint32_t *rk = rk_buf;
	int i = 0;
	uint32_t temp;

	rk[0] = GETU32(enc_key     );
	rk[1] = GETU32(enc_key +  4);
	rk[2] = GETU32(enc_key +  8);
	rk[3] = GETU32(enc_key + 12);
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
	rk[4] = GETU32(enc_key + 16);
	rk[5] = GETU32(enc_key + 20);
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
	rk[6] = GETU32(enc_key + 24);
	rk[7] = GETU32(enc_key + 28);
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
/*
__global__ void computeHMAC_SHA1_AES(
        uint8_t *input_buf, uint8_t *output, 
        size_t *input_size_arr, size_t *output_size_arr,
        int N, uint8_t *checkbits_d,
        int                     *key_idxs,
        struct hmac_sa_entry    *hmac_aes_key_array,
        int32_t                 *offsets)
*/
__global__ void computeHMAC_SHA1_AES(
		uint8_t* input_buf, uint8_t *output_buf,
        size_t *input_size_arr, size_t *output_size_arr,
        int N, uint8_t *checkbits_d,
        const uint8_t* __restrict__ ivs,
        const int32_t* __restrict__ key_idxs, const struct hmac_aes_sa_entry* __restrict__ hmac_aes_key_array,
        const int32_t* __restrict__ offsets)
{
	/* computer the thread id */
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (idx < N) {
        /* Locate data */
        const uint8_t *in  = input_buf + offsets[idx];
        uint8_t *out       = output_buf + offsets[idx];

    /*
        int temp = key_idxs[idx];
        assert(temp == key_array[temp].entry_idx);
        assert(key_array[temp].aes_key != NULL);
    */

        __shared__ uint32_t shared_Te0[256];
        __shared__ uint32_t shared_Te1[256];
        __shared__ uint32_t shared_Te2[256];
        __shared__ uint32_t shared_Te3[256];
        __shared__ uint32_t shared_Rcon[10];

        /* initialize T boxes */
        for (unsigned i = 0 ; i *blockDim.x < 256 ; i++) {
            unsigned index = threadIdx.x + i * blockDim.x;
            if (index >= N)
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
    //	__syncthreads();

        const uint8_t   *key = (const uint8_t*) hmac_aes_key_array[key_idxs[idx]].aes_key;
        uint8_t 		*ivec	= (uint8_t*) (idx * AES_BLOCK_SIZE + ivs); 

        /* Encrypt using cbc mode */
        unsigned long len = (unsigned long) input_size_arr[idx];
        const unsigned char *iv = ivec;

        while (len >= AES_BLOCK_SIZE) {
            *((uint64_t*)out)       = *((uint64_t*)in)       ^ *((uint64_t*)iv);
            *(((uint64_t*)out) + 1) = *(((uint64_t*)in) + 1) ^ *(((uint64_t*)iv) + 1);

            AES_128_encrypt(out, out, key,
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
            AES_128_encrypt(out, out, key,
                    shared_Te0, shared_Te1, shared_Te2, shared_Te3, shared_Rcon);
            iv = out;
        }

        *((uint4*)ivec) = *((uint4*)iv);
        
    //    __syncthreads();

        // HMAC-SHA1 hashing
        int32_t offset = offsets[idx];
        char *hmac_key = (char *) hmac_aes_key_array[key_idxs[idx]].hmac_key; 
        uint16_t length = (uint16_t) input_size_arr[idx];
        if (offset != -1) {
            // printf("TID:%4d \t Offset %10u, Length %10u\n", idx, offset, length);
            HMAC_SHA1((uint32_t*) (input_buf + offset), (uint32_t*) (output_buf + idx * SHA_DIGEST_LENGTH), length, (char*)hmac_key);
            // output_size_arr[idx] = SHA_DIGEST_LENGTH; // as output_roi is CUSTOMDATA, output_size_arr is not used.
        }
    }

	__syncthreads();
	if (threadIdx.x == 0 && checkbits_d != 0)
		*(checkbits_d + blockIdx.x) = 1;
}


/* Among AES_cbc_128_decryption, AES_cbc_128_encryption,
 * AES_ecb_128_encryption and AES_decrypt_key_prepare(),
 * AES_cbc_128_encrypt_gpu() is only used in NBA, for now. */
 
void *nba::ipsec_hmac_sha1_aes_get_cuda_kernel() {
	return reinterpret_cast<void *> (computeHMAC_SHA1_AES);
}

