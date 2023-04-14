#pragma once
#include <cstdio>

#define RED "\x1B[31m"
#define GREEN "\x1B[32m"
#define INFO "\x1B[34m"
#define YELLOW "\x1B[33m"
#define ADDITIONAL "\x1B[35m"
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
    printf("%s... ", msg);
    Start();
  }
  void StopAndPrint() {
    Stop();
    printf("%f ms\n" NORMAL, elapsed());
  }
  float elapsed() {
    float time;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return time;
  }
};

#define TIMER_START(msg) timer.StartAndPrint(msg)
#define TIMER_STOP() timer.StopAndPrint()
#define SET_COLOR(color) printf(color)