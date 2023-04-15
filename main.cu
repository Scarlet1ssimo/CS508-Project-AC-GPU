#include "ac_cpu.h"
#include "kernel.h"
#include "utility.h"
#include <cstdio>
#include <cstdlib>
#include <map>
using std::map;

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char* file, const int line) {
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s at %s:%d", cudaGetErrorString(result), file, line);
    exit(-1);
  }
}
Timer timer;
enum KernelAvailable {
  KERNEL_SIMPLE,
  KERNEL_SHARED_MEM,
  KERNEL_COALESCED_MEM_READ,
};
std::map<KernelAvailable, const char*> kernelName = {
    {KERNEL_SIMPLE, "KERNEL_SIMPLE"},
    {KERNEL_SHARED_MEM, "KERNEL_SHARED_MEM"},
    {KERNEL_COALESCED_MEM_READ, "KERNEL_COALESCED_MEM_READ"},
};
struct AdditionalTestConfig {
  unsigned int randomSeed;
  bool ReorderTrie;
};

// M: pattern length (like 8 for pattern "ACGTACGT")
// N: number of all patterns, typically 16000
// L: text length, typically 1e6
// Charset (4 for ACGT)
// Kernel version
void eval(int M, int N, int L, KernelAvailable kernel_id, const int charSetSize, const char* testName, AdditionalTestConfig config) {
  // Generate random patterns and text
  printf("====================================\n");
  printf("Start testing " YELLOW "%s" NORMAL ": " INFO "(pattern length = %d, %d patterns, text length = %d, |CharSet|=%d)\n" NORMAL,
         testName, M, N, L, charSetSize);
  printf("Using kernel: " YELLOW "%s\n" NORMAL, kernelName[kernel_id]);
  if (config.randomSeed != 0) {
    printf(ADDITIONAL "Using random seed = %u\n" NORMAL, config.randomSeed);
    srand(config.randomSeed);
  }

  char** patterns = (char**) malloc(N * sizeof(char*));
  for (int i = 0; i < N; i++) {
    patterns[i] = (char*) malloc((M + 1) * sizeof(char));
    random_string(patterns[i], charSetSize, M);
  }
  char* text = (char*) malloc((L + 1) * sizeof(char));
  random_string(text, charSetSize, L);

  // Allocate memory on CPU for building Trie and Aho-Corasick Algorithm
  TIMER_START("Building Trie and Aho-Corasick Automaton on CPU");
  int trieNodeBound = N * M;
  int* tr           = (int*) malloc(trieNodeBound * charSetSize * sizeof(int)); // tr[trieNodeBound][charSetSize] -> trieNodeBound;
  int* idx          = (int*) malloc(N * sizeof(int));                           // idx[N] -> trieNodeBound;
  memset(tr, 0, trieNodeBound * charSetSize * sizeof(int));
  int trieNodeNumber = TrieBuildCPU(patterns, tr, idx, M, N, charSetSize);
  if (config.ReorderTrie) {
    printf(ADDITIONAL "\nReordering Trie...");
    TrieReorder(tr, idx, N, trieNodeNumber, charSetSize);
  }
  int* fail      = (int*) malloc(trieNodeNumber * sizeof(int)); // fail[trieNodeNumber] -> trieNodeNumber;
  int* postOrder = (int*) malloc(trieNodeNumber * sizeof(int)); // postOrder[trieNodeNumber] -> trieNodeNumber;
  int* occur_gpu = (int*) malloc(trieNodeNumber * sizeof(int)); // occur[trieNodeNumber] -> L;
  int* occur_cpu = (int*) malloc(trieNodeNumber * sizeof(int)); // occur[trieNodeNumber] -> L;
  memset(fail, 0, trieNodeNumber * sizeof(int));
  memset(occur_cpu, 0, trieNodeNumber * sizeof(int));

  // Build Aho-Corasick Automaton on CPU
  ACBuildCPU(tr, fail, postOrder, charSetSize);
  TIMER_STOP();

  // Setup for GPU launch
  int *d_tr, *d_occur;
  char* d_text;
  TIMER_START("Allocating and copying memory on GPU");
  CUDA_RUNTIME(cudaMalloc(&d_tr, trieNodeNumber * charSetSize * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc(&d_occur, trieNodeNumber * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc(&d_text, (L + 1) * sizeof(char)));
  CUDA_RUNTIME(cudaMemcpy(d_tr, tr, trieNodeNumber * charSetSize * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemcpy(d_text, text, (L + 1) * sizeof(char), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemset(d_occur, 0, trieNodeNumber * sizeof(int)));
  TIMER_STOP();

  // Run Aho-Corasick Algorithm on GPU
  TIMER_START(INFO "Running Aho-Corasick Algorithm on GPU");
  SET_COLOR(YELLOW);
  if (kernel_id == KERNEL_SIMPLE)
    ACGPUSimpleLaunch(d_tr, d_text, d_occur, M, L, charSetSize);
  else if (kernel_id == KERNEL_SHARED_MEM)
    ACGPUSharedMemLaunch(d_tr, d_text, d_occur, M, L, charSetSize, trieNodeNumber);
  else if (kernel_id == KERNEL_COALESCED_MEM_READ)
    ACGPUCoalecedMemReadLaunch(d_tr, d_text, d_occur, M, L, charSetSize);
  else {
    printf(RED "Error: kernel_id = %d is not supported", kernel_id);
    TIMER_STOP();
    // goto bad;
  }
  CUDA_RUNTIME(cudaDeviceSynchronize());
  TIMER_STOP();

  // Copy result back to CPU
  TIMER_START("Copying result back to CPU");
  CUDA_RUNTIME(cudaMemcpy(occur_gpu, d_occur, trieNodeNumber * sizeof(int), cudaMemcpyDeviceToHost));
  TIMER_STOP();
  // End of GPU

  // Run Aho-Corasick Algorithm on CPU
  TIMER_START(INFO "Running Aho-Corasick Algorithm on CPU");
  ACCPU(tr, text, occur_cpu, L, charSetSize);
  TIMER_STOP();
  // Post processing to get the final result
  TIMER_START("Postprocessing on CPU");
  ACPostCPU(occur_cpu, fail, postOrder, trieNodeNumber - 1);
  ACPostCPU(occur_gpu, fail, postOrder, trieNodeNumber - 1);
  TIMER_STOP();

  // Verification against CPU result
  TIMER_START("Verifying result");
  for (int i = 0; i < N; i++)
    if (occur_cpu[idx[i]] != occur_gpu[idx[i]]) {
      printf(RED "Error at %d: %d %d\n", i, occur_cpu[idx[i]], occur_gpu[idx[i]]);
      TIMER_STOP();
      goto bad;
    }
  TIMER_STOP();
  printf(GREEN "Pass\n" NORMAL);
bad:

  printf("====================================\n");
  CUDA_RUNTIME(cudaFree(d_tr));
  CUDA_RUNTIME(cudaFree(d_occur));
  CUDA_RUNTIME(cudaFree(d_text));
  free(tr);
  free(idx);
  free(fail);
  free(postOrder);
  free(occur_gpu);
  free(occur_cpu);
  free(text);
  for (int i = 0; i < N; i++)
    free(patterns[i]);
  free(patterns);
}
int main() {
  srand(time(NULL));
  // eval(8, 16000, 1e6, 0, 4, "ACSimple");
  // eval(8, 16000, 1e7, 0, 4, "ACSimple");
  eval(8, 16000, 1e8, KERNEL_SIMPLE, 4, "ACSimple", {23333, false});
  eval(8, 16000, 1e8, KERNEL_COALESCED_MEM_READ, 4, "ACCoalecedMemRead", {23333, false});
  // eval(8, 16000, 1e8, 1, 4, "ACSharedMem");
  // eval(8, 160, 1e8, 0, 4, "ACSimple");
  eval(8, 16000, 1e8, KERNEL_SHARED_MEM, 4, "ACSharedMem", {23333, false});
  eval(8, 16000, 1e8, KERNEL_SHARED_MEM, 4, "ACSharedMemWithReordering", {23333, true});
  // eval(8, 16000, 1e9, 0, 4, "ACSimple");
}
