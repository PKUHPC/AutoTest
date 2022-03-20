#pragma once

#include <cuda.h>
#include <cstdint>

namespace hice {

class BenchmarkCUDA {
 public:
  BenchmarkCUDA() = delete;

  template<typename Func>
  static float bench(int n_iters, const Func& func) {
    if (!initialized_) {
      cudaEventCreate(&start_);
      cudaEventCreate(&end_);
      func(); // warm up
      initialized_ = true;
    }
    cudaEventRecord(start_, 0);
    cudaDeviceSynchronize();
    for (int i = 1; i <= n_iters; ++i) {
      func();
    }
    cudaEventRecord(end_, 0);
    cudaEventSynchronize(end_);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_, end_);
    return elapsedTime / n_iters;
  }

 private:
  static bool initialized_;
  static cudaEvent_t start_;
  static cudaEvent_t end_;
  // int64_t g_flops_processed_;
};

bool BenchmarkCUDA::initialized_ = false;
cudaEvent_t BenchmarkCUDA::start_;
cudaEvent_t BenchmarkCUDA::end_;

}  // namespace hice

