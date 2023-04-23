#include "kernel.h"
#include <cstddef>
#include <cstdio>

__global__ void CompactText(const char* text, char* text_conpact, const int L) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < L)
    text_conpact[idx] = text[idx];
}

__global__ void CompactText(const char* text, int4x2_t* text_conpact, const int L) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  text_conpact[idx].h0 = (idx * 2 < L) ? text[idx * 2] : 0;
  text_conpact[idx].h1 = (idx * 2 + 1 < L) ? text[idx * 2 + 1] : 0;
}

__global__ void CompactText(const char* text, int2x4_t* text_conpact, const int L) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  text_conpact[idx].q0 = (idx * 4 < L) ? text[idx * 4] : 0;
  text_conpact[idx].q1 = (idx * 4 + 1 < L) ? text[idx * 4 + 1] : 0;
  text_conpact[idx].q2 = (idx * 4 + 2 < L) ? text[idx * 4 + 2] : 0;
  text_conpact[idx].q3 = (idx * 4 + 3 < L) ? text[idx * 4 + 3] : 0;
}
