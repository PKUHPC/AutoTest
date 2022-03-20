#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/core/sparse_tensor.h"
#include "hice/math/matmul.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"

namespace hice {

#if 1
TEST(MatmulCSRTestFloat, SPMV) {
  // csr_a
  std::vector<int64_t> dims_a = {4, 6};
  std::vector<int32_t> indices_a = { 0,  0,  1,  1,  2,  3,  3, 3,
                                     1,  4,  0,  5,  2,  1,  3, 5};
  std::vector<float> values_a = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<int> column_indices_a = {1,  4,  0,  5,  2,  1, 3, 5};
  std::vector<int> row_offsets_a = {0, 2, 4, 5, 8};
  // cpu
  // tensor_b
  std::vector<int64_t> dims_b = {6};
  Tensor sparse_a = wrap_sparse(dims_a, indices_a.data(), values_a.data(),
                                values_a.size(), dtype(kFloat));
  Tensor csr_a = wrap_csr(dims_a, column_indices_a.data(),
                          row_offsets_a.data(), values_a.data(), values_a.size(),
                          device(kCPU).dtype(kFloat), true);
  Tensor tensor_b = rand_uniform(dims_b, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor coo_output = matmul(sparse_a, tensor_b);
  Tensor csr_output = matmul(csr_a, tensor_b);
  ExpectEqualDenseRegardlessDevice(coo_output, csr_output);
#if 1
  // gpu
  Tensor sparse_a_gpu = sparse_a.to(kCUDA);
  Tensor csr_a_gpu = csr_a.to(kCUDA);
  Tensor tensor_b_gpu = tensor_b.to(kCUDA);
  // check consistency
  Tensor coo_output_gpu = matmul(sparse_a_gpu, tensor_b_gpu);
  Tensor csr_output_gpu = matmul(csr_a_gpu, tensor_b_gpu);
  ExpectEqualDenseRegardlessDevice(csr_output_gpu, csr_output);
  ExpectEqualDenseRegardlessDevice(csr_output_gpu, coo_output_gpu);
  // check correctness
  Tensor csr_output_b = matmul(csr_a.to(kDense), tensor_b);
  ExpectEqualDense(csr_output, csr_output_b);
#endif
}
#endif

#if 1
TEST(MatmulCSRTestFloat, SPMM) {
  // sparse_a
  std::vector<int64_t> dims_a = {4, 6};
  std::vector<int32_t> indices_a = { 0,  0,  1,  1,  2,  3,  3, 3,
                                     1,  4,  0,  5,  2,  1,  3, 5};
  std::vector<float> values_a = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<int> column_indices_a = {1,  4,  0,  5,  2,  1, 3, 5};
  std::vector<int> row_offsets_a = {0, 2, 4, 5, 8};
  // tensor_b
  std::vector<int64_t> dims_b = {6,4};
  // cpu
  Tensor sparse_a = wrap_sparse(dims_a, indices_a.data(), values_a.data(),
                                values_a.size(), dtype(kFloat));
  Tensor csr_a = wrap_csr(dims_a, column_indices_a.data(),
                          row_offsets_a.data(), values_a.data(), values_a.size(),
                          device(kCPU).dtype(kFloat), true);
  Tensor tensor_b = rand_uniform(dims_b, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor coo_output = matmul(sparse_a, tensor_b);
  Tensor csr_output = matmul(csr_a, tensor_b);
  ExpectEqualDenseRegardlessDevice(coo_output, csr_output);
#if 1
  // gpu
  Tensor sparse_a_gpu = sparse_a.to(kCUDA);
  Tensor csr_a_gpu = csr_a.to(kCUDA);
  Tensor tensor_b_gpu = tensor_b.to(kCUDA);
  // check consistency
  Tensor coo_output_gpu = matmul(sparse_a_gpu, tensor_b_gpu);
  Tensor csr_output_gpu = matmul(csr_a_gpu, tensor_b_gpu);
  ExpectEqualDenseRegardlessDevice(csr_output, csr_output_gpu);
  ExpectEqualDenseRegardlessDevice(csr_output_gpu, coo_output_gpu);
  // check correctness
  Tensor csr_output_b = matmul(csr_a.to(kDense), tensor_b);
  ExpectEqualDense(csr_output, csr_output_b);
#endif
}
#endif


#if 1
TEST(MatmulCSRTestFloat, SPGEMM) {
  // sparse_a
  std::vector<int64_t> dims_a = {4, 6};
  std::vector<int32_t> indices_a = { 0,  0,  1,  1,  2,  3,  3, 3,
                                     1,  4,  5,  0,  2,  1,  3, 5};
  std::vector<float> values_a =    {10, 20, 40, 30, 50, 60, 70, 70};
  Tensor sparse_a = wrap_sparse(dims_a, indices_a.data(), values_a.data(),
                                values_a.size(), dtype(kFloat));
  // sparse_b
  std::vector<int64_t> dims_b = {6, 4};
  std::vector<int32_t> indices_b = { 0,  1,  1,  2,  3,  4,  5, 5,
                                     1,  0,  3,  2,  3,  0,  1, 3};
  std::vector<float> values_b =    {10, 20, 30, 40, 50, 60, 70, 70};
  Tensor sparse_b = wrap_sparse(dims_b, indices_b.data(), values_b.data(),
                                values_b.size(), dtype(kFloat));
 // csr_a & csr_b
  std::vector<int> column_indices_a = {1,  4,  5,  0,  2,  1, 3, 5};
  std::vector<int> row_offsets_a = {0, 2, 4, 5, 8};
  Tensor csr_a = wrap_csr(dims_a, column_indices_a.data(),
                          row_offsets_a.data(), values_a.data(), values_a.size(),
                          device(kCPU).dtype(kFloat), true);
  std::vector<int> column_indices_b = {1,  0,  3,  2,  3,  0,  1, 3};
  std::vector<int> row_offsets_b = {0, 1, 3, 4, 5, 6, 8};
  Tensor csr_b = wrap_csr(dims_b, column_indices_b.data(),
                          row_offsets_b.data(), values_b.data(), values_b.size(),
                          device(kCPU).dtype(kFloat), true);
  // output
  std::vector<int64_t> dims_output = {4, 4};
  // cpu
  Tensor coo_output = matmul(sparse_a, sparse_b);
  Tensor csr_output = matmul(csr_a, csr_b);
  ExpectEqualDenseRegardlessDevice(coo_output.values(), csr_output.values());
  // TODO: compare indicies of coo_output and csr_output
#if 1
  // gpu
  Tensor sparse_a_gpu = sparse_a.to(kCUDA);
  Tensor sparse_b_gpu = sparse_b.to(kCUDA);
  Tensor csr_a_gpu = csr_a.to(kCUDA);
  Tensor csr_b_gpu = csr_b.to(kCUDA);
  // check consistency
  Tensor coo_output_gpu = matmul(sparse_a_gpu, sparse_b_gpu);
  Tensor csr_output_gpu = matmul(csr_a_gpu, csr_b_gpu);
  ExpectEqualSparseRegardlessDevice(csr_output, csr_output_gpu);   // check coo
  ExpectEqualDenseRegardlessDevice(coo_output.values(), csr_output_gpu.values());
  // check correctness
  Tensor csr_output_b = matmul(csr_a.to(kDense), csr_b.to(kDense));
  ExpectEqualDense(csr_output.to(kDense), csr_output_b);
#endif
}
#endif


} // namespace hice
