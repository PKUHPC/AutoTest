#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/ml/knn.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/tools/compare.h"

// #include <time.h>
// #include <sys/time.h>

namespace hice {
using ::testing::Each;

TEST(KnnTest, DenseFloat) {
  // const int num_of_refs = 32 * 1024;
  // const int num_of_query = 5;
  // const int num_of_feature = 8192;

  const int num_of_refs = 8 * 512;
  const int num_of_query = 5;
  const int num_of_feature = 256;
  int k = 16;

  // struct timeval start;
  // struct timeval end;
  // unsigned long dur_serial, dur_parallel;

  Tensor ref = rand_uniform({num_of_refs, num_of_feature}, 0, 10.0, device(kCPU).dtype(kFloat));
  Tensor query = rand_uniform({num_of_query, num_of_feature}, 0, 10.0, device(kCPU).dtype(kFloat));
  Tensor labels = rand_uniform({num_of_refs, 1}, 0, 5, device(kCPU).dtype(kInt32));
  TensorPrinter tp;

  // gettimeofday(&start,NULL);
  Tensor result = knn(ref, labels, query, k);
  // gettimeofday(&end, NULL);
  // dur_serial = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  // std::cout << "Serial Knn " << (double)dur_serial / 1000000 << "s." << std::endl;
  // tp.print(result);

  Tensor ref_cuda = ref.to(kCUDA);
  Tensor query_cuda = query.to(kCUDA);
  Tensor labels_cuda = labels.to(kCUDA);
  // gettimeofday(&start,NULL);
  Tensor result_cuda = knn(ref_cuda, labels_cuda, query_cuda, k);
  // gettimeofday(&end, NULL);
  // dur_parallel = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  // std::cout << "Parallel Knn " << (double)dur_parallel / 1000000 << "s." << std::endl;
  // tp.print(result_cuda);
  
  // std::cout << "Speedup = " << (double)dur_serial / dur_parallel << "." << std::endl;
  ExpectEqualDenseRegardlessDevice(result, result_cuda);

}
}  // namespace hice