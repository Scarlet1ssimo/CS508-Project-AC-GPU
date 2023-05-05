#pragma once
#include "compact.h"
#include "kernel_impl.h"
#include "utility.h"

template <int charSetSize>
void ACGPUSimpleLaunch(const int* tr, const unsigned char* text, int* occur, const int M, const int L) {
  const int TILE_SIZE  = 32;
  const int BLOCK_SIZE = 512;
  int blockNum         = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;
  ACGPUSimple<charSetSize, TILE_SIZE, BLOCK_SIZE><<<blockNum, BLOCK_SIZE>>>(tr, text, occur, M, L);
}

// #define PROFILING
template <int charSetSize>
void ACGPUSharedMemLaunch(const int* tr, const unsigned char* text, int* occur, const int M, const int L, const int trieNodeNumber) {
  const int TILE_SIZE        = 32;
  const int BLOCK_SIZE       = 512;
  const int GPUbinSize       = 1024; // could be tuned for control divergence vs shared memory usage
  const int MaxSharedMemSize = 49152;
  int blockNum               = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;
#ifndef PROFILING
  if (trieNodeNumber * sizeof(int) < MaxSharedMemSize)
    ACGPUSharedMemBranchFree<charSetSize, TILE_SIZE, BLOCK_SIZE, GPUbinSize>
        <<<blockNum, BLOCK_SIZE, trieNodeNumber * sizeof(int)>>>(tr, text, occur, M, L, trieNodeNumber);
  else
    ACGPUSharedMem<charSetSize, TILE_SIZE, BLOCK_SIZE, GPUbinSize><<<blockNum, BLOCK_SIZE>>>(tr, text, occur, M, L, trieNodeNumber);
#else
  unsigned long long h_data[2], *branchCnt;
  cudaMalloc(&branchCnt, sizeof(h_data));
  cudaMemset(branchCnt, 0, sizeof(h_data));

  ACGPUSharedMemProfiling<charSetSize, TILE_SIZE, BLOCK_SIZE, GPUbinSize>
      <<<blockNum, BLOCK_SIZE>>>(tr, text, occur, M, L, trieNodeNumber, branchCnt);

  cudaMemcpy(h_data, branchCnt, sizeof(h_data), cudaMemcpyDeviceToHost);
  printf("\nBin portion:%d/%d=%.2lf\n", GPUbinSize, trieNodeNumber, 1.0 * GPUbinSize / trieNodeNumber);
  printf("Branch portion:%llu/%llu=%.2lf\n", h_data[0], h_data[0] + h_data[1], 1.0 * h_data[0] / (h_data[0] + h_data[1]));
#endif
}

template <int charSetSize>
void ACGPUCoalecedMemReadLaunch(const int* tr, const unsigned char* text, int* occur, const int M, const int L) {
  const int TILE_SIZE  = 64;
  const int BLOCK_SIZE = 32;
  int blockNum         = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;
  ACGPUCoalecedMemRead<charSetSize, TILE_SIZE, BLOCK_SIZE>
      <<<blockNum, BLOCK_SIZE, (TILE_SIZE * BLOCK_SIZE + M - 1) * sizeof(int)>>>(tr, text, occur, M, L);
}

__global__ void CompactText(const unsigned char* text, unsigned char* text_conpact, const int L);
__global__ void CompactText(const unsigned char* text, int4x2_t* text_conpact, const int L);
__global__ void CompactText(const unsigned char* text, int2x4_t* text_conpact, const int L);
__global__ void CompactText(const unsigned char* text, int1x8_t* text_conpact, const int L);

template <int charSetSize, typename T>
void ACGPUCompactMemLaunch(const int* tr, const unsigned char* text, T* text_compact, int* occur, int M, int L) {
  const int numElement = SizeTraits<charSetSize>::numElement;
  const int TILE_SIZE  = 32;
  const int BLOCK_SIZE = 512;

  if (numElement > 1) {
    // Compressible text, compact text
    CompactText<<<ceil(L * 1.0 / (BLOCK_SIZE * numElement)), BLOCK_SIZE>>>(text, text_compact, L);
    cudaDeviceSynchronize();
    ACGPUCompactMem<charSetSize, TILE_SIZE>
        <<<ceil(L * 1.0 / (BLOCK_SIZE * TILE_SIZE * numElement)), BLOCK_SIZE>>>(tr, text_compact, occur, M, L);
  } else {
    // Incompressible text, trigger the kernel directly
    ACGPUCompactMem<charSetSize, TILE_SIZE><<<ceil(L * 1.0 / (BLOCK_SIZE * TILE_SIZE * numElement)), BLOCK_SIZE>>>(tr, text, occur, M, L);
  }
}

template <int charSetSize, typename T>
void ACGPUEqLengthLaunch(const int* tr, const unsigned char* text, T* text_compact, int* occur, int N, int M, int L, int trieNodeNumber) {
  const int TILE_SIZE        = 32;
  const int BLOCK_SIZE       = 512;
  const int GPUbinSize       = 1024; // could be tuned for control divergence vs shared memory usage
  const int MaxSharedMemSize = 49152;
  int blockNum               = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;

  ACGPUEqLength<charSetSize, TILE_SIZE, BLOCK_SIZE><<<blockNum, BLOCK_SIZE, N * sizeof(int)>>>(tr, text, occur, N, M, L, trieNodeNumber);
}
