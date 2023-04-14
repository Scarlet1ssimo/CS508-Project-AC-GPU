#include "kernel.hu"
#include <cstdio>
const int TILE_SIZE  = 32;
const int BLOCK_SIZE = 1024;
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
// shared memory
// coalesced memory fetch into shared memory
// int4
void ACGPUSimpleLaunch(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize) {
  int blockNum = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;
  ACGPUSimple<<<blockNum, BLOCK_SIZE>>>(tr, text, occur, M, L, charSetSize);
}
const int GPUbinSize = 2048;
__global__ void ACGPUSharedMem(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize,
                               const int trieNodeNumber) {
  int idx         = blockIdx.x * blockDim.x * TILE_SIZE;
  int threadStart = idx + threadIdx.x * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + M - 1;

  int state = 0;
  __shared__ int localOccur[GPUbinSize];
  for (int i = threadIdx.x; i < GPUbinSize; i += blockDim.x) {
    localOccur[i] = 0;
  }
  __syncthreads();
  for (int i = threadStart; i < threadEnd && i < L; i++) {
    state = tr[state * charSetSize + text[i]];
    if (state < GPUbinSize) // CPU:Reorder trie node index by depth
      atomicAdd(&localOccur[state], 1);
    else
      atomicAdd(&occur[state], 1);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < GPUbinSize; i += blockDim.x) {
    atomicAdd(&occur[i], localOccur[i]);
  }
}
void ACGPUSharedMemLaunch(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize,
                          const int trieNodeNumber) {
  int blockNum = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;
  ACGPUSharedMem<<<blockNum, BLOCK_SIZE>>>(tr, text, occur, M, L, charSetSize, trieNodeNumber);
}
/*
0816......1917....
threadstart
      threadend
------|------|------|------|------|------|------|
 t1   |t2    |t3    |t4    |t5    |t6    |t7    |...
0-----8------16-------block-----------------------
*/