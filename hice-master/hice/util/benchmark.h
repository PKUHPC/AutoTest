#pragma once

#include <sys/time.h>

namespace hice {
  
static const int kDefaultTrails = 50;

template<typename Fn>
HICE_API float bench_cpu(int n_trails, const Fn& fn) {
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

template<typename Fn>
HICE_API float bench_cpu(const Fn& fn) {
  return bench_cpu(kDefaultTrails, fn);
}

}  // namespace hice