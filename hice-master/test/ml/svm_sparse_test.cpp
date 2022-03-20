#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/ml/svm.h"

#include "gtest/gtest.h"
#include "test/tools/compare.h"

namespace hice {
TEST(SvmSparseTest, svm_sparse) {
  TensorPrinter tp;
  SvmParam param;
  long int num_vects = 6;
  // long int num_vects = 5;
  // long int num_vects = 4;
  long int dim_vects = 3;
  // std::vector<float> values = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
  // std::vector<int> column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  // std::vector<int> row_offsets = {0, 3, 6, 9, 12};
  // std::vector<float> values = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5};
  // std::vector<int> column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  // std::vector<int> row_offsets = {0, 3, 6, 9, 12, 15};
  std::vector<float> values = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8};
  std::vector<int> column_indices = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<int> row_offsets = {0, 3, 6, 9, 12, 15, 18};
  Tensor train_data_csr = wrap_csr({num_vects, dim_vects}, column_indices.data(),
                    row_offsets.data(), values.data(), values.size(),  device(kCPU).dtype(kFloat), true);
  Tensor label = full({num_vects}, 1, device(kCPU).dtype(kInt32));
  Tensor predict_data_csr = wrap_csr({num_vects, dim_vects}, column_indices.data(),
                    row_offsets.data(), values.data(), values.size(),  device(kCPU).dtype(kFloat), true);
  Tensor result = full({num_vects}, 1, device(kCPU).dtype(kInt32));

  int *label_ptr = label.mutable_data<int>();
  for (int i = 0; i < num_vects / 2; i++)
    label_ptr[i] = 1;
  for (int i = num_vects / 2; i < num_vects; i++)
    label_ptr[i] = -1;

  Tensor cuda_train_data = train_data_csr.to(kCUDA);
  Tensor cuda_predict_data = predict_data_csr.to(kCUDA);
  Tensor cuda_train_data_dense = cuda_train_data.to(kDense);
  Tensor cuda_predict_data_dense = cuda_predict_data.to(kDense);
  Tensor cuda_label = label.to(kCUDA);
  Tensor cuda_result = result.to(kCUDA);
  Tensor cuda_result_dense = result.to(kCUDA);

//inplace
  svm(train_data_csr, label, predict_data_csr, result, param);
  svm(cuda_train_data, cuda_label, cuda_predict_data, cuda_result, param);
  svm(cuda_train_data_dense, cuda_label, cuda_predict_data_dense, cuda_result_dense, param);
//outplace
  Tensor result_outplace = svm(train_data_csr, label, predict_data_csr, param);
  Tensor cuda_result_outplace = svm(cuda_train_data, cuda_label, cuda_predict_data, param);
  Tensor cuda_result_dense_outplace = svm(cuda_train_data_dense, cuda_label, cuda_predict_data_dense, param);
  ExpectEqualDenseRegardlessDevice(result, cuda_result);
  ExpectEqualDenseRegardlessDevice(result, cuda_result_dense);
  ExpectEqualDenseRegardlessDevice(result, result_outplace);
  ExpectEqualDenseRegardlessDevice(result, cuda_result_outplace);
  ExpectEqualDenseRegardlessDevice(result, cuda_result_dense_outplace);
}
// #endif
} // namespace hice
