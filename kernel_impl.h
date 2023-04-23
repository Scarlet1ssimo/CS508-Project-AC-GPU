#pragma once

template <int charSetSize, int TILE_SIZE, int BLOCK_SIZE>
__global__ void ACGPUSimple(const int* tr, const char* text, int* occur, const int M, const int L) {
  int idx         = blockIdx.x * blockDim.x * TILE_SIZE;
  int threadStart = idx + threadIdx.x * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + M - 1;

  int state = 0;
  for (int i = threadStart; i < threadEnd && i < L; i++) {
    state = tr[state * charSetSize + text[i]];
    atomicAdd(&occur[state], 1); // Optimizable
  }
}

template <int charSetSize, int TILE_SIZE, int BLOCK_SIZE>
__global__ void ACGPUCoalecedMemRead(const int* tr, const char* text, int* occur, const int M, const int L) {
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

template <int charSetSize, int TILE_SIZE, int BLOCK_SIZE, int GPUbinSize>
__global__ void ACGPUSharedMem(const int* tr, const char* text, int* occur, const int M, const int L,
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

template <int charSetSize, int TILE_SIZE, int BLOCK_SIZE, int GPUbinSize>
__global__ void ACGPUSharedMemProfiling(const int* tr, const char* text, int* occur, const int M, const int L,
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

template <int charSetSize, int TILE_SIZE>
__global__ void ACGPUCompactMem(const int* tr, const char* text_conpact, int* occur, const int M, const int L) {
  int idx         = blockIdx.x * blockDim.x * TILE_SIZE;
  int threadStart = idx + threadIdx.x * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + M - 1;

  int state = 0;
  for (int i = threadStart; i < threadEnd && i < L; i++) {
    state = tr[state * charSetSize + text_conpact[i]];
    atomicAdd(&occur[state], 1); // Optimizable
  }
}

template <int charSetSize, int TILE_SIZE>
__global__ void ACGPUCompactMem(const int* tr, const int4x2_t* text_conpact, int* occur, const int M, const int L) {
  int idx         = blockIdx.x * blockDim.x + threadIdx.x;
  int threadStart = idx * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + (M - 1) / 2;

  int state = 0;
  for (int i = threadStart; i < threadEnd && i * 2 < L; i++) {
    state = tr[state * charSetSize + text_conpact[i].h0];
    atomicAdd(&occur[state], 1);
    state = tr[state * charSetSize + text_conpact[i].h1];
    atomicAdd(&occur[state], 1);
    if (i == threadEnd - 1 && threadEnd < L / 2) {
      if (M - 1 % 2 == 1) {
        state = tr[state * charSetSize + text_conpact[threadEnd].h0];
        atomicAdd(&occur[state], 1);
      }
    }
  }
}

template <int charSetSize, int TILE_SIZE>
__global__ void ACGPUCompactMem(const int* tr, const int2x4_t* text_conpact, int* occur, const int M, const int L) {
  int idx         = blockIdx.x * blockDim.x + threadIdx.x;
  int threadStart = idx * TILE_SIZE;
  int threadEnd   = threadStart + TILE_SIZE + (M - 1) / 4;

  int state = 0;
  for (int i = threadStart; i < threadEnd && i < L / 4; i++) {
    state = tr[state * charSetSize + text_conpact[i].q0];
    atomicAdd(&occur[state], 1);
    state = tr[state * charSetSize + text_conpact[i].q1];
    atomicAdd(&occur[state], 1);
    state = tr[state * charSetSize + text_conpact[i].q2];
    atomicAdd(&occur[state], 1);
    state = tr[state * charSetSize + text_conpact[i].q3];
    atomicAdd(&occur[state], 1);
    if (i == threadEnd - 1 && threadEnd < L / 4) {
      if ((M - 1) % 4 >= 1) {
        state = tr[state * charSetSize + text_conpact[threadEnd].q0];
        atomicAdd(&occur[state], 1);
      }
      if ((M - 1) % 4 >= 2) {
        state = tr[state * charSetSize + text_conpact[threadEnd].q1];
        atomicAdd(&occur[state], 1);
      }
      if ((M - 1) % 4 >= 3) {
        state = tr[state * charSetSize + text_conpact[threadEnd].q2];
        atomicAdd(&occur[state], 1);
      }
    }
  }
}

