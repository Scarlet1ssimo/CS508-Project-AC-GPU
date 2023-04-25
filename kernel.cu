#include "kernel.h"
#include <cstdio>

__global__ void CompactText(const unsigned char* text, unsigned char* text_conpact, const int L) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < L)
    text_conpact[idx] = text[idx];
}

__global__ void CompactText(const unsigned char* text, int4x2_t* text_conpact, const int L) {
  int idx              = blockIdx.x * blockDim.x + threadIdx.x;
  text_conpact[idx].h0 = (idx * 2 < L) ? text[idx * 2] : 0;
  text_conpact[idx].h1 = (idx * 2 + 1 < L) ? text[idx * 2 + 1] : 0;
}

__global__ void CompactText(const unsigned char* text, int2x4_t* text_conpact, const int L) {
  int idx              = blockIdx.x * blockDim.x + threadIdx.x;
  text_conpact[idx].q0 = (idx * 4 < L) ? text[idx * 4] : 0;
  text_conpact[idx].q1 = (idx * 4 + 1 < L) ? text[idx * 4 + 1] : 0;
  text_conpact[idx].q2 = (idx * 4 + 2 < L) ? text[idx * 4 + 2] : 0;
  text_conpact[idx].q3 = (idx * 4 + 3 < L) ? text[idx * 4 + 3] : 0;
}

__global__ void CompactText(const unsigned char* text, int1x8_t* text_conpact, const int L) {
  int idx              = blockIdx.x * blockDim.x + threadIdx.x;
  text_conpact[idx].o0 = (idx * 8 < L) ? text[idx * 8] : 0;
  text_conpact[idx].o1 = (idx * 8 + 1 < L) ? text[idx * 8 + 1] : 0;
  text_conpact[idx].o2 = (idx * 8 + 2 < L) ? text[idx * 8 + 2] : 0;
  text_conpact[idx].o3 = (idx * 8 + 3 < L) ? text[idx * 8 + 3] : 0;
  text_conpact[idx].o4 = (idx * 8 + 4 < L) ? text[idx * 8 + 4] : 0;
  text_conpact[idx].o5 = (idx * 8 + 5 < L) ? text[idx * 8 + 5] : 0;
  text_conpact[idx].o6 = (idx * 8 + 6 < L) ? text[idx * 8 + 6] : 0;
  text_conpact[idx].o7 = (idx * 8 + 7 < L) ? text[idx * 8 + 7] : 0;
}
