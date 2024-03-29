#include "ac_cpu.h"
#include "kernel.h"
#include "utility.h"
#include <cstdio>
#include <ctime>
#include <map>
#include <vector>
using std::map;

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char* file, const int line) {
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s at %s:%d", cudaGetErrorString(result), file, line);
    exit(-1);
  }
}
Timer timer;
enum KernelAvailable { KERNEL_SIMPLE, KERNEL_SHARED_MEM, KERNEL_COALESCED_MEM_READ, KERNEL_COMPACT_MEM, KERNEL_EQ_LENGTH };
std::map<KernelAvailable, const char*> kernelName = {{KERNEL_SIMPLE, "KERNEL_SIMPLE"},
                                                     {KERNEL_SHARED_MEM, "KERNEL_SHARED_MEM"},
                                                     {KERNEL_COALESCED_MEM_READ, "KERNEL_COALESCED_MEM_READ"},
                                                     {KERNEL_COMPACT_MEM, "KERNEL_COMPACT_MEM"},
                                                     {KERNEL_EQ_LENGTH, "KERNEL_EQ_LENGTH"}};

// M: pattern length (like 8 for pattern "ACGTACGT")
// N: number of all patterns, typically 16000
// L: text length, typically 1e6
// Charset (4 for ACGT)
// Kernel version
template <int charSetSize>
void eval(int M, int N, int L, KernelAvailable kernel_id, const char* testName, AdditionalTestConfig config) {
  // Generate random patterns and text
  printf("====================================\n");
  printf("Start testing " YELLOW "%s" NORMAL ": " INFO "(pattern length = %d, %d patterns, text length = %d, |CharSet|=%d)\n" NORMAL,
         testName, M, N, L, charSetSize);
  printf("Using kernel: " YELLOW "%s\n" NORMAL, kernelName[kernel_id]);
  if (config.randomSeed != 0) {
    printf(ADDITIONAL "Using random seed = %u\n" NORMAL, config.randomSeed);
    srand(config.randomSeed);
  }
  if (SizeTraits<charSetSize>::numElement == 1 && kernel_id == KERNEL_COMPACT_MEM) {
    printf(ADDITIONAL "Detected using compact memory for incompressible charSet.\n" NORMAL);
  }

  unsigned char **patterns, *text;
  if (config.CI != nullptr) {
    TIMER_START("Parsing file from config...");
    if (!parseDataFrom(M, N, L, charSetSize, &patterns, &text, *config.CI)) {
      printf("Failed to parse data from config file. Abort.\n");
      return;
    }
  } else {
    TIMER_START("Generating random patterns and text...");
    randomData(M, N, L, charSetSize, &patterns, &text);
    if (config.dumpDataOnly && charSetSize == 4) {
      TIMER_STOP();
      TIMER_START("Dump and return...");
      dumpACGTData(M, N, L, patterns, text);
      TIMER_STOP();
      return;
    }
  }
  TIMER_STOP();

  // Allocate memory on CPU for building Trie and Aho-Corasick Algorithm
  TIMER_START("Building Trie and Aho-Corasick Automaton on CPU");
  int trieNodeBound = N * M;
  int* tr           = (int*) malloc(trieNodeBound * charSetSize * sizeof(int)); // tr[trieNodeBound][charSetSize] -> trieNodeBound;
  int* idx          = (int*) malloc(N * sizeof(int));                           // idx[N] -> trieNodeBound;
  memset(tr, 0, trieNodeBound * charSetSize * sizeof(int));
  memset(idx, 0, N * sizeof(int));
  int trieNodeNumber = TrieBuildCPU(patterns, tr, idx, M, N, charSetSize);
  int occurGPUSlots  = kernel_id == KERNEL_EQ_LENGTH ? N : trieNodeNumber;
  int occurStart     = trieNodeNumber - occurGPUSlots;
  printf("Getting %d Trie nodes... ", trieNodeNumber);
  if (kernel_id == KERNEL_EQ_LENGTH) {
    printf(ADDITIONAL "\nReordering Trie...");
    TrieReorder(tr, idx, N, trieNodeNumber, charSetSize);
  } else if (config.ReorderTrie) {
    printf(ADDITIONAL "\nReordering Trie...");
    TrieReorder(tr, idx, N, trieNodeNumber, charSetSize);
  }
  int* fail      = (int*) malloc(trieNodeNumber * sizeof(int)); // fail[trieNodeNumber] -> trieNodeNumber;
  int* postOrder = (int*) malloc(trieNodeNumber * sizeof(int)); // postOrder[trieNodeNumber] -> trieNodeNumber;
  int* occur_gpu = (int*) malloc(occurGPUSlots * sizeof(int));  // occur[trieNodeNumber] -> L;
  int* occur_cpu = (int*) malloc(trieNodeNumber * sizeof(int)); // occur[trieNodeNumber] -> L;
  memset(fail, 0, trieNodeNumber * sizeof(int));
  memset(postOrder, 0, trieNodeNumber * sizeof(int));
  memset(occur_gpu, 0, occurGPUSlots * sizeof(int));
  memset(occur_cpu, 0, trieNodeNumber * sizeof(int));

  // Build Aho-Corasick Automaton on CPU
  ACBuildCPU(tr, fail, postOrder, charSetSize);
  TIMER_STOP();

  // Setup for GPU launch
  int *d_tr, *d_occur;
  unsigned char* d_text;
  using T = typename SizeTraits<charSetSize>::elementTy;
  T* d_text_compact;
  TIMER_START("Allocating and copying memory on GPU");
  CUDA_RUNTIME(cudaMalloc(&d_tr, trieNodeNumber * charSetSize * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc(&d_occur, occurGPUSlots * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc(&d_text, (L + 1) * sizeof(unsigned char)));
  if (kernel_id == KERNEL_COMPACT_MEM) {
    printf(ADDITIONAL "Also for compact text... " NORMAL);
    CUDA_RUNTIME(cudaMalloc(&d_text_compact, (L + 1) * sizeof(T)));
  }
  CUDA_RUNTIME(cudaMemcpy(d_tr, tr, trieNodeNumber * charSetSize * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemcpy(d_text, text, (L + 1) * sizeof(unsigned char), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemset(d_occur, 0, occurGPUSlots * sizeof(int)));
  TIMER_STOP();

  // Run Aho-Corasick Algorithm on GPU
  TIMER_START(INFO "Running Aho-Corasick Algorithm on GPU");
  SET_COLOR(YELLOW);
  if (kernel_id == KERNEL_SIMPLE)
    ACGPUSimpleLaunch<charSetSize>(d_tr, d_text, d_occur, M, L);
  else if (kernel_id == KERNEL_SHARED_MEM)
    ACGPUSharedMemLaunch<charSetSize>(d_tr, d_text, d_occur, M, L, trieNodeNumber);
  else if (kernel_id == KERNEL_COALESCED_MEM_READ)
    ACGPUCoalecedMemReadLaunch<charSetSize>(d_tr, d_text, d_occur, M, L);
  else if (kernel_id == KERNEL_COMPACT_MEM)
    ACGPUCompactMemLaunch<charSetSize, T>(d_tr, d_text, d_text_compact, d_occur, M, L);
  else if (kernel_id == KERNEL_EQ_LENGTH)
    ACGPUEqLengthLaunch<charSetSize, T>(d_tr, d_text, d_text_compact, d_occur, N, M, L, trieNodeNumber);
  else {
    printf(RED "Error: kernel_id = %d is not supported", kernel_id);
    TIMER_STOP();
    goto bad;
  }
  CUDA_RUNTIME(cudaDeviceSynchronize());
  TIMER_STOP();

  // Copy result back to CPU
  TIMER_START("Copying result back to CPU");
  CUDA_RUNTIME(cudaMemcpy(occur_gpu, d_occur, occurGPUSlots * sizeof(int), cudaMemcpyDeviceToHost));
  TIMER_STOP();
  // End of GPU

  // Run Aho-Corasick Algorithm on CPU
  if (!config.skipCPUVerify) {
    TIMER_START(INFO "Running Aho-Corasick Algorithm on CPU");
    ACCPU(tr, text, occur_cpu, L, charSetSize);
    TIMER_STOP();
  }
  // Post processing to get the final result
  TIMER_START("Postprocessing on CPU");
  if (kernel_id != KERNEL_EQ_LENGTH)
    ACPostCPU(occur_gpu, fail, postOrder, trieNodeNumber - 1);
  TIMER_STOP();

  if (!config.skipCPUVerify) {
    // Verification against CPU result
    TIMER_START("Verifying result");
    ACPostCPU(occur_cpu, fail, postOrder, trieNodeNumber - 1);
    for (int i = 0; i < N; i++)
      if (occur_cpu[idx[i]] != occur_gpu[idx[i] - occurStart]) {
        printf(RED "Error at %d: %d %d\n", i, occur_cpu[idx[i]], occur_gpu[idx[i] - occurStart]);
        TIMER_STOP();
        goto bad;
      }
    TIMER_STOP();
    printf(GREEN "Pass\n" NORMAL);
  }
bad:

  printf("====================================\n");
  CUDA_RUNTIME(cudaFree(d_tr));
  CUDA_RUNTIME(cudaFree(d_occur));
  CUDA_RUNTIME(cudaFree(d_text));
  if (kernel_id == KERNEL_COMPACT_MEM) {
    CUDA_RUNTIME(cudaFree(d_text_compact));
  }
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
  // CustomInput input1 = {
  //     // Found from https://github.com/kaushiksk/boyer-moore-aho-corasick-in-cuda.git
  //     "dataset/boyer-moore-aho-corasick-in-cuda/keywords.txt",
  //     "dataset/boyer-moore-aho-corasick-in-cuda/data.txt",
  // };
  // // Found from https://github.com/Abhi9k/AhoCorasickParallel.git
  // CustomInput input2l = {
  //     "dataset/cuda_pattern_matching/keywords.txt",
  //     "dataset/cuda_pattern_matching/large.txt",
  // };
  // CustomInput input2m = {
  //     "dataset/cuda_pattern_matching/keywords.txt",
  //     "dataset/cuda_pattern_matching/medium.txt",
  // };
  // CustomInput input2s = {
  //     "dataset/cuda_pattern_matching/keywords.txt",
  //     "dataset/cuda_pattern_matching/small.txt",
  // };

  const int C = 4, L = 1 << 27;

  std::vector<int> Ns = {1000, 4000, 8000, 12000, 16000};
  std::vector<int> Ms = {8};

  for (int N : Ns) {
    for (int M : Ms) {
      // eval<C>(M, N, L, KERNEL_SIMPLE, "ACSimple", {23333, false, nullptr, true}); //dump only
      printf("# Patterns = %d, Pattern Length = %d\n", N, M);
      // eval<C>(M, N, L, KERNEL_SIMPLE, "ACSimple", {23333, false});
      // eval<C>(M, N, L, KERNEL_COALESCED_MEM_READ, "ACCoalecedMemRead", {23333, false});
      eval<C>(M, N, L, KERNEL_EQ_LENGTH, "SpecialOptmizationIfPatternEqualLength", {23333, false});
      eval<C>(M, N, L, KERNEL_SHARED_MEM, "ACSharedMem", {23333, true});
      // eval<C>(M, N, L, KERNEL_COMPACT_MEM, "ACCompactMem", {23333, true});
    }
  }
}
