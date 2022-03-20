#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/core/sparse_tensor.h"
#include "hice/math/matmul.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"

namespace hice {

#if 1

TEST(MatmulSparseTestFloat, SPMV) {
  // sparse_a
  std::vector<int64_t> dims_a = {4, 6};
  std::vector<int32_t> indices_a = { 0,  0,  1,  1,  2,  3,  3, 3,
                                     1,  4,  0,  5,  2,  1,  3, 5};
  std::vector<float> values_a =    {10, 20, 30, 40, 50, 60, 70, 80};
  // tensor_b
  std::vector<int64_t> dims_b = {6};
  // cpu
  Tensor sparse_a = wrap_sparse(dims_a, indices_a.data(), values_a.data(), 
                                values_a.size(), dtype(kFloat));
  Tensor tensor_b = rand_uniform(dims_b, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor output = matmul(sparse_a, tensor_b);
  // gpu
  Tensor sparse_a_gpu = sparse_a.to(kCUDA);
  Tensor tensor_b_gpu = tensor_b.to(kCUDA);

  // check consistency
  Tensor output_gpu = matmul(sparse_a_gpu, tensor_b_gpu);
  ExpectEqualDenseRegardlessDevice(output, output_gpu);
  // check correctness
  Tensor output_b = matmul(sparse_a.to(kDense), tensor_b);
  ExpectEqualDense(output, output_b);
}
#endif

#if 1
TEST(MatmulSparseTestFloat, SPMM) {
  // sparse_a
  std::vector<int64_t> dims_a = {4, 6};
  std::vector<int32_t> indices_a = { 0,  0,  1,  1,  2,  3,  3, 3,
                                     1,  4,  0,  5,  2,  1,  3, 5};
  std::vector<float> values_a =    {10, 20, 30, 40, 50, 60, 70, 80};
  // tensor_b
  std::vector<int64_t> dims_b = {6, 4};

  Tensor sparse_a = wrap_sparse(dims_a, indices_a.data(), values_a.data(), 
                                values_a.size(), dtype(kFloat));
  Tensor tensor_b = rand_uniform(dims_b, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor output = matmul(sparse_a, tensor_b);
  // gpu
  Tensor sparse_a_gpu = sparse_a.to(kCUDA);
  Tensor tensor_b_gpu = tensor_b.to(kCUDA);
  
  // check consistency
  Tensor output_gpu = matmul(sparse_a_gpu, tensor_b_gpu);
  ExpectEqualDenseRegardlessDevice(output, output_gpu);
  // check correctness
  Tensor output_b = matmul(sparse_a.to(kDense), tensor_b);
  ExpectEqualDense(output, output_b);
}
#endif

#if 1
TEST(MatmulSparseTestFloat, SPGEMM) {
  // sparse_a
  std::vector<int64_t> dims_a = {4, 6};
  std::vector<int32_t> indices_a = { 0,  0,  1,  1,  2,  3,  3, 3,
                                     1,  4,  0,  5,  2,  1,  3, 5};
  std::vector<float> values_a =    {10, 20, 30, 40, 50, 60, 70, 80};
  Tensor sparse_a = wrap_sparse(dims_a, indices_a.data(), values_a.data(), 
                                values_a.size(), dtype(kFloat));
  // sparse_b
  std::vector<int64_t> dims_b = {6, 4};
  std::vector<int32_t> indices_b = { 0,  1,  1,  2,  3,  4,  5, 5,
                                     1,  0,  3,  2,  3,  0,  1, 3};
  std::vector<float> values_b =    {10, 20, 30, 40, 50, 60, 70, 80};
  Tensor sparse_b = wrap_sparse(dims_b, indices_b.data(), values_b.data(), 
                                values_b.size(), dtype(kFloat));
  // output
  std::vector<int64_t> dims_output = {4, 4};
  // cpu
  Tensor output = matmul(sparse_a, sparse_b);
  // gpu
  Tensor sparse_a_gpu = sparse_a.to(kCUDA);
  Tensor sparse_b_gpu = sparse_b.to(kCUDA);
  
  // check consistency
  Tensor output_gpu = matmul(sparse_a_gpu, sparse_b_gpu);
  ExpectEqualSparseRegardlessDevice(output, output_gpu);
  // check correctness
  Tensor output_b = matmul(sparse_a.to(kDense), sparse_b.to(kDense));
  ExpectEqualDense(output.to(kDense), output_b);
}
#endif


} // namespace hice

