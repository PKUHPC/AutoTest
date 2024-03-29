// This file is based on aten\src\Aten\Parallel.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage. 
#pragma once
#include <atomic>
#include <cstddef>
#include <exception>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hice {
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants paralellism. For example, when summing an array, it is
// deemed inefficient to parallelise over arrays shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr int64_t GRAIN_SIZE = 32768;

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline int get_max_threads() {
#ifdef _OPENMP
  // std::cout<<"omp_get_max_threads()="<<omp_get_max_threads()<<std::endl;
  return omp_get_max_threads();
#else
  HICE_LOG(WARNING) << "_OPENMP not defined!";
  return 1;
#endif
}

inline int get_thread_num() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

inline bool in_parallel() {
#ifdef _OPENMP
  return omp_in_parallel();
#else
  return false;
#endif
}

template <typename TFunc>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const TFunc& func) {
#ifdef _OPENMP
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
#pragma omp parallel if (!omp_in_parallel() && ((end - begin) >= grain_size))
  {
    // std::cout<<"tid = "<<omp_get_thread_num()<<std::endl;
    int64_t num_threads = omp_get_num_threads();
    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      try {
        func(begin_tid, std::min(end, chunk_size + begin_tid));
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
#else
  if (begin < end) {
    func(begin, end);
  }
#endif
}

/*
parallel_reduce
begin: index at which to start applying reduction
end: index at which to stop applying reduction
grain_size: number of elements per chunk. impacts number of elements in
intermediate results tensor and degree of parallelization.
ident: identity for binary combination function sf. sf(ident, x) needs to return
x.
f: function for reduction over a chunk. f needs to be of signature scalar_t
f(int64_t partial_begin, int64_t partial_end, scalar_t identifiy)
sf: function to combine two partial results. sf needs to be of signature
scalar_t sf(scalar_t x, scalar_t y)
For example, you might have a tensor of 10000 entires and want to sum together
all the elements. Parallel_reduce with a grain_size of 2500 will then allocate
an intermediate result tensor with 4 elements. Then it will execute the function
"f" you provide and pass the beginning and end index of these chunks, so
0-2499, 2500-4999, etc. and the combination identity. It will then write out
the result from each of these chunks into the intermediate result tensor. After
that it'll reduce the partial results from each chunk into a single number using
the combination function sf and the identity ident. For a total summation this
would be "+" and 0 respectively. This is similar to tbb's approach [1], where
you need to provide a function to accumulate a subrange, a function to combine
two partial results and an identity.
[1] https://software.intel.com/en-us/node/506154
*/
// template <class scalar_t, class F, class SF>
// inline scalar_t parallel_reduce(
//     const int64_t begin,
//     const int64_t end,
//     const int64_t grain_size,
//     const scalar_t ident,
//     const F f,
//     const SF sf) {
//   if (get_num_threads() == 1) {
//     return f(begin, end, ident);
//   } else {
//     const int64_t num_results = divup((end - begin), grain_size);
//     std::vector<scalar_t> results(num_results);
//     scalar_t* results_data = results.data();
// #pragma omp parallel for if ((end - begin) >= grain_size)
//     for (int64_t id = 0; id < num_results; id++) {
//       int64_t i = begin + id * grain_size;
//       results_data[id] = f(i, i + std::min(end - i, grain_size), ident);
//     }
//     return std::accumulate(
//         results_data, results_data + results.size(), ident, sf);
//   }
// }

} // namespace hice