#include "ac_cpu.h"
#include "kernel.hu"
#include <cstdio>

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char* file, const int line) {
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s at %s:%d", cudaGetErrorString(result), file, line);
    exit(-1);
  }
}

// M: pattern length (like 8 for pattern "ACGTACGT")
// N: number of all patterns, typically 16000
// L: text length, typically 1e6
// Charset (4 for ACGT)
// Kernel version
void eval(int M, int N, int L, int kernel_id, const int charSetSize, const char* testName) {
  // TODO: Prettify the output
  // At least output the run time

  // Generate random patterns and text
  printf("====================================\n");
  printf("Start testing %s: (M=%d, N=%d, L=%d)\n", testName, M, N, L);
  char** patterns = (char**) malloc(N * sizeof(char*));
  for (int i = 0; i < N; i++) {
    patterns[i] = (char*) malloc((M + 1) * sizeof(char));
    random_string(patterns[i], charSetSize, M);
  }
  char* text = (char*) malloc((L + 1) * sizeof(char));
  random_string(text, charSetSize, L);

  // Allocate memory on CPU for building Trie and Aho-Corasick Algorithm
  int trieNodeBound  = N * M;
  int* tr            = (int*) malloc(trieNodeBound * charSetSize * sizeof(int)); // tr[trieNodeBound][charSetSize] -> trieNodeBound;
  int* idx           = (int*) malloc(N * sizeof(int));                           // idx[N] -> trieNodeBound;
  memset(tr, 0, trieNodeBound * charSetSize * sizeof(int));
  int trieNodeNumber = TrieBuildCPU(patterns, tr, idx, M, N, charSetSize);
  int* fail          = (int*) malloc(trieNodeNumber * sizeof(int)); // fail[trieNodeNumber] -> trieNodeNumber;
  int* postOrder     = (int*) malloc(trieNodeNumber * sizeof(int)); // postOrder[trieNodeNumber] -> trieNodeNumber;
  int* out_cpu       = (int*) malloc(trieNodeNumber * sizeof(int)); // out[N] -> L;
  int* out_gpu       = (int*) malloc(trieNodeNumber * sizeof(int)); // out[N] -> L;
  int* occur_gpu     = (int*) malloc(trieNodeNumber * sizeof(int)); // occur[trieNodeNumber] -> L;
  int* occur_cpu     = (int*) malloc(trieNodeNumber * sizeof(int)); // occur[trieNodeNumber] -> L;
  memset(fail, 0, trieNodeNumber * sizeof(int));
  memset(out_cpu, 0, trieNodeNumber * sizeof(int));
  memset(out_gpu, 0, trieNodeNumber * sizeof(int));
  memset(occur_cpu, 0, trieNodeNumber * sizeof(int));

  // Build Aho-Corasick Automaton on CPU
  ACBuildCPU(tr, fail, postOrder, charSetSize);

  // Setup for GPU launch
  int *d_tr, *d_occur;
  char* d_text;
  CUDA_RUNTIME(cudaMalloc(&d_tr, trieNodeNumber * charSetSize * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc(&d_occur, trieNodeNumber * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc(&d_text, (L + 1) * sizeof(char)));
  CUDA_RUNTIME(cudaMemcpy(d_tr, tr, trieNodeNumber * charSetSize * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemcpy(d_text, text, (L + 1) * sizeof(char), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemset(d_occur, 0, trieNodeNumber * sizeof(int)));

  // Run Aho-Corasick Algorithm on GPU
  if (kernel_id == 0)
    ACGPUSimpleLaunch(d_tr, d_text, d_occur, M, L, charSetSize);
  // TODO: Add more kernel launch here
  else {
    printf("Error: kernel_id = %d is not supported", kernel_id);
    goto bad;
  }

  // Copy result back to CPU
  CUDA_RUNTIME(cudaDeviceSynchronize());
  CUDA_RUNTIME(cudaMemcpy(occur_gpu, d_occur, trieNodeNumber * sizeof(int), cudaMemcpyDeviceToHost));
  // End of GPU

  // Run Aho-Corasick Algorithm on CPU
  ACCPU(tr, text, occur_cpu, L, charSetSize);
  // Post processing to get the final result
  ACPostCPU(out_cpu, fail, postOrder, trieNodeNumber - 1);
  ACPostCPU(out_gpu, fail, postOrder, trieNodeNumber - 1);

  // Verification against CPU result
  for (int i = 0; i < N; i++)
    if (out_cpu[idx[i]] != out_gpu[idx[i]]) {
      printf("Error at %d: %d %d\n", i, out_cpu[i], out_gpu[i]);
      goto bad;
    }
  printf("Pass\n");
bad:
  printf("====================================\n");
  CUDA_RUNTIME(cudaFree(d_tr));
  CUDA_RUNTIME(cudaFree(d_occur));
  CUDA_RUNTIME(cudaFree(d_text));
  free(tr);
  free(idx);
  free(fail);
  free(postOrder);
  free(out_cpu);
  free(out_gpu);
  free(occur_gpu);
  free(occur_cpu);
  free(text);
  for (int i = 0; i < N; i++)
    free(patterns[i]);
  free(patterns);
}
int main() {
  srand(940012978);
  eval(8, 16000, 1e6, 0, 4, "ACSimple");
  eval(8, 16000, 1e7, 0, 4, "ACSimple");
  eval(8, 16000, 1e8, 0, 4, "ACSimple");
  eval(8, 16000, 1e9, 0, 4, "ACSimple");
}
