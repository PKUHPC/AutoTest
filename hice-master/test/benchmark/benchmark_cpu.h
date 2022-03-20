#pragma once

#include <sys/time.h>
#include <cstdint>

namespace hice {

class BenchmarkCPU {
 public:
  BenchmarkCPU() = delete;

  template<typename Func>
  static float bench(int n_iters, const Func& func) {
    gettimeofday(&start_, NULL);
    for (int i = 1; i <= n_iters; ++i) {
      func();
    }
    gettimeofday(&end_, NULL);
    float elapsedTime = (end_.tv_sec - start_.tv_sec) * 1e6 + (end_.tv_usec - start_.tv_usec);
    return elapsedTime / n_iters;
  }

 private:
  static timeval start_;
  static timeval end_;
  // int64_t g_flops_processed_;
};

timeval BenchmarkCPU::start_ = {0, 0};
timeval BenchmarkCPU::end_ = {0, 0};

}  // namespace hice

