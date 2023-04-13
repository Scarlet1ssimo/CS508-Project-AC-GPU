#include "kernel.hu"
const int TILE_SIZE  = 32;
const int BLOCK_SIZE = 1024;
__global__ void ACGPUSimple(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize) {
  int idx         = blockIdx.x * blockDim.x;
  int threadStart = idx + threadIdx.x * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + M - 1;

  int state = 0;
  for (int i = threadStart; i < threadEnd; i++) {
    state = tr[state * charSetSize + text[i]];
    atomicAdd(&occur[state], 1); // Optimizable
  }
}
void ACGPUSimpleLaunch(const int* tr, const char* text, int* occur, const int M, const int L, const int charSetSize) {
  int blockNum = (L - 1) / (BLOCK_SIZE * TILE_SIZE) + 1;
  ACGPUSimple<<<blockNum, BLOCK_SIZE>>>(tr, text, occur, M, L, charSetSize);
}
