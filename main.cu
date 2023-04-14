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
#define RED "\x1B[31m"
#define GREEN "\x1B[32m"
#define INFO "\x1B[34m"
#define YELLOW "\x1B[33m"
#define NORMAL "\033[0m"
struct Timer {
  cudaEvent_t start, stop;
  Timer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  ~Timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void Start() {
    cudaEventRecord(start, 0);
  }
  void Stop() {
    cudaEventRecord(stop, 0);
  }
  void StartAndPrint(const char* msg) {
    printf("%s...", msg);
    Start();
  }
  void StopAndPrint() {
    Stop();
    printf(" %f ms\n" NORMAL, elapsed());
  }
  float elapsed() {
    float time;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return time;
  }
};
// M: pattern length (like 8 for pattern "ACGTACGT")
// N: number of all patterns, typically 16000
// L: text length, typically 1e6
// Charset (4 for ACGT)
// Kernel version
Timer timer;
#define TIMER_START(msg) timer.StartAndPrint(msg)
#define TIMER_STOP() timer.StopAndPrint()
#define SET_COLOR(color) printf(color)

void eval(int M, int N, int L, int kernel_id, const int charSetSize, const char* testName) {
  // TODO: Prettify the output
  // At least output the run time

  // Generate random patterns and text
  printf("====================================\n");
  printf(YELLOW "Start testing %s: (M=%d, N=%d, L=%d)\n" NORMAL, testName, M, N, L);
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
  int* fail          = (int*) malloc(trieNodeNumber * sizeof(int)); // fail[trieNodeNumber] -> trieNodeNumber;
  int* postOrder     = (int*) malloc(trieNodeNumber * sizeof(int)); // postOrder[trieNodeNumber] -> trieNodeNumber;
  int* occur_gpu     = (int*) malloc(trieNodeNumber * sizeof(int)); // occur[trieNodeNumber] -> L;
  int* occur_cpu     = (int*) malloc(trieNodeNumber * sizeof(int)); // occur[trieNodeNumber] -> L;
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
  if (kernel_id == 0)
    ACGPUSimpleLaunch(d_tr, d_text, d_occur, M, L, charSetSize);
  else if (kernel_id == 1) {
    ACGPUSharedMemLaunch(d_tr, d_text, d_occur, M, L, charSetSize, trieNodeNumber);
  } else {
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
  srand(940012978);
  // eval(8, 16000, 1e6, 0, 4, "ACSimple");
  // eval(8, 16000, 1e7, 0, 4, "ACSimple");
  eval(8, 16000, 1e8, 0, 4, "ACSimple");
  // eval(8, 16000, 1e8, 1, 4, "ACSharedMem");
  // eval(8, 160, 1e8, 0, 4, "ACSimple");
  eval(8, 16000, 1e8, 1, 4, "ACSharedMem");
  // eval(8, 16000, 1e9, 0, 4, "ACSimple");
}
