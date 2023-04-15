#include "kernel.h"
#include <cstddef>
#include <cstdio>

template <int TILE_SIZE, int BLOCK_SIZE>
__global__ void ACGPUSimple(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize) {
  int idx         = blockIdx.x * blockDim.x * TILE_SIZE;
  int threadStart = idx + threadIdx.x * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + M - 1;

  int state = 0;
  for (int i = threadStart; i < threadEnd && i < L; i++) {
    state = tr[state * charSetSize + text[i]];
    atomicAdd(&occur[state], 1); // Optimizable
  }
}

void ACGPUSimpleLaunch(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize) {
  const int TILE_SIZE  = 32;
  const int BLOCK_SIZE = 512;
  int blockNum         = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;
  ACGPUSimple<TILE_SIZE, BLOCK_SIZE><<<blockNum, BLOCK_SIZE>>>(tr, text, occur, M, L, charSetSize);
}

template <int TILE_SIZE, int BLOCK_SIZE>
__global__ void ACGPUCoalecedMemRead(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize) {
  int idx         = blockIdx.x * blockDim.x * TILE_SIZE;
  int threadStart = threadIdx.x * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + M - 1;

  extern __shared__ char localText[];
  for (int i = threadIdx.x; i < blockDim.x * TILE_SIZE + M - 1; i += blockDim.x)
    localText[i] = text[idx + i];
  __syncthreads();

  int state = 0;
  for (int i = threadStart; i < threadEnd && idx + i < L; i++) {
    state = tr[state * charSetSize + localText[i]];
    atomicAdd(&occur[state], 1); // Optimizable
  }
}

void ACGPUCoalecedMemReadLaunch(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize) {
  const int TILE_SIZE  = 64;
  const int BLOCK_SIZE = 32;
  int blockNum         = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;
  ACGPUCoalecedMemRead<TILE_SIZE, BLOCK_SIZE>
      <<<blockNum, BLOCK_SIZE, (TILE_SIZE * BLOCK_SIZE + M - 1) * sizeof(int)>>>(tr, text, occur, M, L, charSetSize);
}

template <int TILE_SIZE, int BLOCK_SIZE, int GPUbinSize>
__global__ void ACGPUSharedMem(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize,
                               const int trieNodeNumber) {
  int idx         = blockIdx.x * blockDim.x * TILE_SIZE;
  int threadStart = idx + threadIdx.x * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + M - 1;

  __shared__ int localOccur[GPUbinSize];
  for (int i = threadIdx.x; i < GPUbinSize; i += blockDim.x)
    localOccur[i] = 0;
  __syncthreads();
  int state = 0;
  for (int i = threadStart; i < threadEnd && i < L; i++) {
    state = tr[state * charSetSize + text[i]];
    if (state < GPUbinSize) // CPU:Reorder trie node index by depth(frequency)
      atomicAdd(&localOccur[state], 1);
    else
      atomicAdd(&occur[state], 1);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < GPUbinSize; i += blockDim.x)
    atomicAdd(&occur[i], localOccur[i]);
}

template <int TILE_SIZE, int BLOCK_SIZE, int GPUbinSize>
__global__ void ACGPUSharedMemProfiling(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize,
                                        const int trieNodeNumber, unsigned long long* branchCnt) {
  int idx         = blockIdx.x * blockDim.x * TILE_SIZE;
  int threadStart = idx + threadIdx.x * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + M - 1;

  __shared__ int localOccur[GPUbinSize];
  for (int i = threadIdx.x; i < GPUbinSize; i += blockDim.x)
    localOccur[i] = 0;
  __syncthreads();
  int state = 0;
  for (int i = threadStart; i < threadEnd && i < L; i++) {
    state = tr[state * charSetSize + text[i]];
    if (state < GPUbinSize) {
      atomicAdd(&branchCnt[0], 1);
      atomicAdd(&localOccur[state], 1);
    } else {
      atomicAdd(&branchCnt[1], 1);
      atomicAdd(&occur[state], 1);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < GPUbinSize; i += blockDim.x)
    atomicAdd(&occur[i], localOccur[i]);
}

// #define PROFILING
void ACGPUSharedMemLaunch(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize,
                          const int trieNodeNumber) {
  const int TILE_SIZE  = 32;
  const int BLOCK_SIZE = 512;
  const int GPUbinSize = 1024;
  int blockNum         = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;
#ifndef PROFILING
  ACGPUSharedMem<TILE_SIZE, BLOCK_SIZE, GPUbinSize><<<blockNum, BLOCK_SIZE>>>(tr, text, occur, M, L, charSetSize, trieNodeNumber);
#else
  unsigned long long h_data[2], *branchCnt;
  cudaMalloc(&branchCnt, sizeof(h_data));
  cudaMemset(branchCnt, 0, sizeof(h_data));

  ACGPUSharedMemProfiling<TILE_SIZE, BLOCK_SIZE, GPUbinSize>
      <<<blockNum, BLOCK_SIZE>>>(tr, text, occur, M, L, charSetSize, trieNodeNumber, branchCnt);

  cudaMemcpy(h_data, branchCnt, sizeof(h_data), cudaMemcpyDeviceToHost);
  printf("\nBin portion:%d/%d=%.2lf\n", GPUbinSize, trieNodeNumber, 1.0 * GPUbinSize / trieNodeNumber);
  printf("Branch portion:%llu/%llu=%.2lf\n", h_data[0], h_data[0] + h_data[1], 1.0 * h_data[0] / (h_data[0] + h_data[1]));
#endif
}

// shared memory
// coalesced memory fetch into shared memory
// int4