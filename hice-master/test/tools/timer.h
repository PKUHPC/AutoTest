#pragma once

#include <cuda.h>
#include <sys/time.h>

#define NUM_RUN 10

struct cuda_timer {
  cudaEvent_t start_event, stop_event;
  void start() {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
    cudaDeviceSynchronize();
  }
  float stop() {
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_event, stop_event);
    return elapsedTime;
  }
};