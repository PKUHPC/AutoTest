#include "hice/core/sparse_tensor.h"
#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/math/matmul.h"

#include "gtest/gtest.h"
#include "test/tools/compare.h"

namespace hice {

// CSR test
TEST(SparseTensorTest, NewCSR) {
  std::vector<int64_t> dims = {4, 3};
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<int> row_offsets = {0, 3, 6, 9, 12};
  Tensor csr_a = wrap_csr({4, 3}, column_indices.data(),
                    row_offsets.data(), values.data(), values.size(), dtype(kFloat), true);
  Tensor csr_b = new_csr({4, 3}, csr_a.column_indices(), csr_a.row_offsets(), csr_a.values(), dtype(kFloat), true);
  ExpectEqualSparse(csr_a, csr_b);
}
TEST(SparseTensorTest, CSR_Coalesce) {
  // csr_a, coalesced
  std::vector<int64_t> dims = {4, 3};
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<int> row_offsets = {0, 3, 6, 9, 12};
  Tensor csr_a = wrap_csr(dims, column_indices.data(),
                    row_offsets.data(), values.data(), values.size(), dtype(kFloat), true);
  // csr_b_uncoal
  std::vector<float> b_uncoal_values = {3, 2, 1, 4, 5, 6, 7, 8, 9, 12, 11, 10};
  std::vector<int> b_uncoal_column_indices = {2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0};
  std::vector<int> b_uncoal_row_offsets = {0, 3, 6, 9, 12};
  Tensor csr_b_uncoal = wrap_csr(dims, b_uncoal_column_indices.data(),
                    b_uncoal_row_offsets.data(), b_uncoal_values.data(), b_uncoal_values.size(), dtype(kFloat), true);
  // csr_b_uncoal to coalesced
  Tensor csr_b = csr_b_uncoal.to_coalesced();
  ExpectEqualSparse(csr_a, csr_b);
}
TEST(SparseTensorTest, CSR_resize_nnz) {
  // csr_a
  std::vector<int64_t> dims = {4, 3};
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<int> row_offsets = {0, 3, 6, 9, 12};
  Tensor csr_a = wrap_csr(dims, column_indices.data(),
                    row_offsets.data(), values.data(), values.size(), dtype(kFloat), true);
  // csr_b
  std::vector<float> b_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> b_column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
  std::vector<int> b_row_offsets = {0, 3, 6, 9, 10};
  Tensor csr_b = wrap_csr(dims, b_column_indices.data(),
                    b_row_offsets.data(), b_values.data(), b_values.size(), dtype(kFloat), true);
  // resize csr_a with new n_nnz
  int new_n_nnz = b_values.size();
  csr_a.resize_with_nnz(new_n_nnz);
  ExpectEqualDense(csr_a.values(), csr_b.values());
  ExpectEqualDense(csr_a.column_indices(), csr_b.column_indices());
  // ExpectEqualSparse(sparse_a, sparse_b);
}
TEST(SparseTensorTest, CSRToDense) {
  // csr_a
  std::vector<int64_t> dims = {4, 3};
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<int> row_offsets = {0, 3, 6, 9, 12};
  Tensor csr_a = wrap_csr(dims, column_indices.data(),
                    row_offsets.data(), values.data(), values.size(), dtype(kFloat), true);
  Tensor dense_a = csr_a.to(kDense);
  // csr_b
  std::vector<float> b_uncoal_values = {3, 2, 1, 4, 5, 6, 7, 8, 9, 12, 11, 10};
  std::vector<int> b_uncoal_column_indices = {2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0};
  std::vector<int> b_uncoal_row_offsets = {0, 3, 6, 9, 12};
  Tensor csr_b_uncoal = wrap_csr(dims, b_uncoal_column_indices.data(),
                    b_uncoal_row_offsets.data(), b_uncoal_values.data(), b_uncoal_values.size(), dtype(kFloat), true);
  // csr_b_uncoal to coalesced
  Tensor csr_b = csr_b_uncoal.to_coalesced();
  Tensor dense_uncoal_b = csr_b_uncoal.to(kDense);
  Tensor dense_b = csr_b.to(kDense);
  ExpectEqualDense(dense_b, dense_uncoal_b);
  ExpectEqualDense(dense_a, dense_b);
}

TEST(SparseTensorTest, CSRToCOO) {
  // csr_a
  std::vector<int64_t> dims = {4, 3};
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<int> row_offsets = {0, 3, 6, 9, 12};
  Tensor csr_a = wrap_csr(dims, column_indices.data(),
                    row_offsets.data(), values.data(), values.size(), dtype(kFloat), true);
  Tensor coo_a = csr_a.to(kCOO);
  // csr_b
  std::vector<float> b_uncoal_values = {3, 2, 1, 4, 5, 6, 7, 8, 9, 12, 11, 10};
  std::vector<int> b_uncoal_column_indices = {2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0};
  std::vector<int> b_uncoal_row_offsets = {0, 3, 6, 9, 12};
  Tensor csr_b_uncoal = wrap_csr(dims, b_uncoal_column_indices.data(),
                    b_uncoal_row_offsets.data(), b_uncoal_values.data(), b_uncoal_values.size(), dtype(kFloat), true);
  // csr_b_uncoal to coalesced
  Tensor csr_b = csr_b_uncoal.to_coalesced();
  Tensor coo_uncoal_b = csr_b_uncoal.to(kCOO);
  Tensor coo_b = csr_b.to(kCOO);
  ExpectEqualSparse(coo_b, coo_uncoal_b);
  ExpectEqualSparse(coo_a, coo_b);
  // TensorPrinter tp;
  // tp.print(coo_b);
  // tp.print(coo_a);
}
#if 1
TEST(SparseTensorTest, DenseToCSR) {
  // csr_a
  std::vector<int64_t> dims = {4, 3};
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<int> row_offsets = {0, 3, 6, 9, 12};
  Tensor csr_a = wrap_csr(dims, column_indices.data(),
                    row_offsets.data(), values.data(), values.size(), dtype(kFloat), true);
  ExpectEqualSparse(csr_a, csr_a.to(kDense).to(kCSR));
}
#endif

// Sparse(coo) test
TEST(SparseTensorTest, NewSparse) {
  TensorPrinter tp;
  Tensor sparse = new_sparse(dtype(kFloat));
  // tp.print(sparse);
  sparse = new_sparse({2, 5}, dtype(kFloat));
  // tp.print(sparse);
}
TEST(SparseTensorTest, Coalesce) {
  // sparse_a, coalesced
  std::vector<int64_t> dims = {4, 6};
  std::vector<int32_t> indices_a_vec = {0, 0, 1, 1, 2, 3, 3, 3,
                                        1, 4, 0, 5, 2, 1, 3, 5};
  std::vector<float> values_a_vec = {10, 20, 30, 40, 50, 60, 70, 80};
  Tensor sparse_a = wrap_sparse(dims, indices_a_vec.data(), values_a_vec.data(),
                                values_a_vec.size(), dtype(kFloat));
  // sparse_b
  std::vector<int32_t> indices_b_uncoal_vec = {3, 0, 1, 1, 2, 3, 0, 3, 0,
                                               1, 4, 0, 5, 2, 3, 1, 5, 4};
  std::vector<float> values_b_uncoal_vec = {60, 10, 30, 40, 50, 70, 10, 80, 10};
  Tensor sparse_b_uncoal =
      wrap_sparse(dims, indices_b_uncoal_vec.data(), values_b_uncoal_vec.data(),
                  values_b_uncoal_vec.size(), dtype(kFloat));
  Tensor sparse_b = sparse_b_uncoal.to_coalesced();

  ExpectEqualSparse(sparse_a, sparse_b);
}

TEST(SparseTensorTest, Resize_nnz) {
  // sparse_a, coalesced
  std::vector<int64_t> dims = {4, 6};
  std::vector<int32_t> indices_a_vec = {0, 0, 1, 1, 2, 3, 3, 3,
                                        1, 4, 0, 5, 2, 1, 3, 5};
  std::vector<float> values_a_vec = {10, 20, 30, 40, 50, 60, 70, 80};
  Tensor sparse_a = wrap_sparse(dims, indices_a_vec.data(), values_a_vec.data(),
                                values_a_vec.size(), dtype(kFloat));
  // sparse_b, coalesced
  std::vector<int32_t> indices_b_vec = {0, 0, 1, 1, 2, 3, 1, 4, 0, 5, 2, 1};
  std::vector<float> values_b_vec = {10, 20, 30, 40, 50, 60};
  Tensor sparse_b = wrap_sparse(dims, indices_b_vec.data(), values_b_vec.data(),
                                values_b_vec.size(), dtype(kFloat));
  sparse_a.resize_with_nnz(6);
  ExpectEqualSparse(sparse_a, sparse_b);
}

TEST(SparseTensorTest, SparseToDense) {
  // sparse_a, coalesced
  std::vector<int64_t> dims = {4, 6};
  std::vector<int32_t> indices_a_vec = {0, 0, 1, 1, 2, 3, 3, 3,
                                        1, 4, 0, 5, 2, 1, 3, 5};
  std::vector<float> values_a_vec = {10, 20, 30, 40, 50, 60, 70, 80};
  Tensor sparse_a = wrap_sparse(dims, indices_a_vec.data(), values_a_vec.data(),
                                values_a_vec.size(), dtype(kFloat));
  // sparse_b
  std::vector<int32_t> indices_b_uncoal_vec = {3, 0, 1, 1, 2, 3, 0, 3, 0,
                                               1, 4, 0, 5, 2, 3, 1, 5, 4};
  std::vector<float> values_b_uncoal_vec = {60, 10, 30, 40, 50, 70, 10, 80, 10};
  Tensor sparse_b_uncoal =
      wrap_sparse(dims, indices_b_uncoal_vec.data(), values_b_uncoal_vec.data(),
                  values_b_uncoal_vec.size(), dtype(kFloat));
  Tensor sparse_b = sparse_b_uncoal.to_coalesced();
  Tensor dense_a = sparse_a.to(kDense);
  Tensor dense_b = sparse_b.to(kDense);
  ExpectEqualDense(dense_a, dense_b);
}

TEST(SparseTensorTest, DenseToSparse) {
  std::vector<int64_t> dims = {4, 6};
  std::vector<int32_t> indices_a_vec = {0, 0, 1, 1, 2, 3, 3, 3,
                                        1, 4, 0, 5, 2, 1, 3, 5};
  std::vector<float> values_a_vec = {10, 20, 30, 40, 50, 60, 70, 80};
  Tensor sparse_a = wrap_sparse(dims, indices_a_vec.data(), values_a_vec.data(),
                                values_a_vec.size(), dtype(kFloat));
  ExpectEqualSparse(sparse_a, sparse_a.to(kDense).to(kCOO));
}

TEST(SparseTensorTest, SparseToCSR) {
  // sparse_a, coalesced
  std::vector<int64_t> dims = {4, 6};
  std::vector<int32_t> indices_a_vec = {0, 0, 1, 1, 2, 3, 3, 3,
                                        1, 4, 0, 5, 2, 1, 3, 5};
  std::vector<float> values_a_vec = {10, 20, 30, 40, 50, 60, 70, 80};
  Tensor sparse_a = wrap_sparse(dims, indices_a_vec.data(), values_a_vec.data(),
                                values_a_vec.size(), dtype(kFloat));
  // sparse_b
  std::vector<int32_t> indices_b_uncoal_vec = {3, 0, 1, 1, 2, 3, 0, 3, 0,
                                               1, 4, 0, 5, 2, 3, 1, 5, 4};
  std::vector<float> values_b_uncoal_vec = {60, 10, 30, 40, 50, 70, 10, 80, 10};
  Tensor sparse_b_uncoal =
      wrap_sparse(dims, indices_b_uncoal_vec.data(), values_b_uncoal_vec.data(),
                  values_b_uncoal_vec.size(), dtype(kFloat));
  Tensor sparse_b = sparse_b_uncoal.to_coalesced();
  Tensor csr_a = sparse_a.to(kCSR);
  Tensor csr_b = sparse_b.to(kCSR);
  ExpectEqualSparse(csr_a, csr_b);
  // TensorPrinter tp;
  // tp.print(csr_a);
}

}  // namespace hice
