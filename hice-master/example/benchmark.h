#pragma once

#include <sys/time.h>
#include <cstdint>

namespace hice {

static const int kCPUTrails = 500;
static const int kCUDATrails = 500;

class BenchmarkCPU {
 public:
  BenchmarkCPU() = delete;

  template<typename Func>
  static float bench(int n_trails, const Func& fn) {
    // warmup
    fn();
    // timing
    timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (int i = 0; i < n_trails; ++i) {
      fn();
    }
    gettimeofday(&t2, NULL);
    return (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / n_trails;
  }

  template<typename Func>
  static float bench(const Func& fn) {
    return bench(kCPUTrails, fn);
  }
};

class BenchmarkCUDA {
 public:
  BenchmarkCUDA() = delete;

  template<typename Func>
  static float bench(int n_trails, const Func& func) {
    cudaEvent_t start_;
    cudaEvent_t end_;
    cudaEventCreate(&start_);
    cudaEventCreate(&end_);

    func(); // warm up
    cudaEventRecord(start_, 0);
    cudaEventSynchronize(start_);
    for (int i = 1; i <= n_trails; ++i) {
      func();
      cudaDeviceSynchronize();
    }
    cudaEventRecord(end_, 0);
    cudaEventSynchronize(end_);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_, end_);
    return elapsedTime / n_trails;
  }

  template<typename Func>
  static float bench(const Func& fn) {
    return bench(kCUDATrails, fn);
  }
};

}  // namespace hice

