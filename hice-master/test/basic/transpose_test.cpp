#include "hice/core/tensor.h"
#include "hice/core/shape_util.h"
#include "hice/core/index_util.h"
#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/basic/transpose.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;


void TENSOR_EXPECT_EQ(Tensor tensor1, Tensor tensor2) {
  EXPECT_EQ(tensor1.size(), tensor2.size());
  EXPECT_EQ(tensor1.offset(), tensor2.offset());
  EXPECT_EQ(tensor1.data_type(), tensor2.data_type());
  EXPECT_EQ(tensor1.device(), tensor2.device());
  EXPECT_EQ(tensor1.shape(), tensor2.shape());
  EXPECT_EQ(tensor1.strides(), tensor2.strides());
  Tensor tensor1_new = tensor1.device_type() == kCPU ? tensor1 : tensor1.to(kCPU);
  Tensor tensor2_new = tensor2.device_type() == kCPU ? tensor2 : tensor2.to(kCPU);
  auto size = tensor1.size();
  for(int i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(tensor1_new.data<float>()[i], 
                    tensor2_new.data<float>()[i]);
  }
}

void TransposeTestDevice(DeviceType device_type) {

  TensorOptions options = dtype(kFloat).device(device_type);

  Tensor scalar = hice::rand_uniform({}, 1.0, 10.0, options);
  Tensor transposed_scalar = hice::transpose(scalar);
  EXPECT_EQ(transposed_scalar.ndim(), 0);
  EXPECT_EQ(transposed_scalar.size(), scalar.size());
  EXPECT_THAT(transposed_scalar.dims(), IsEmpty());
  EXPECT_THAT(transposed_scalar.layout().minor_to_major(), IsEmpty());
  TENSOR_EXPECT_EQ(scalar, hice::transpose(transposed_scalar));

  Tensor vector = hice::rand_uniform({2}, 1.0, 10.0, options);
  Tensor transposed_vector = hice::transpose(vector);
  EXPECT_EQ(transposed_vector.ndim(), 1);
  EXPECT_EQ(transposed_vector.size(), vector.size());
  EXPECT_THAT(transposed_vector.dims(), ElementsAre(2));
  EXPECT_THAT(transposed_vector.layout().minor_to_major(), ElementsAre(0));
  TENSOR_EXPECT_EQ(vector, hice::transpose(transposed_vector));

  Tensor matrix = hice::rand_uniform({3, 4}, 1.0, 10.0, options);
  Tensor transposed_matrix = hice::transpose(matrix);
  EXPECT_EQ(transposed_matrix.ndim(), 2);
  EXPECT_EQ(transposed_matrix.size(), matrix.size());
  EXPECT_THAT(transposed_matrix.dims(), ElementsAre(4, 3));
  EXPECT_THAT(transposed_matrix.layout().minor_to_major(), ElementsAre(0, 1));
  TENSOR_EXPECT_EQ(matrix, hice::transpose(transposed_matrix));

  Tensor tensor = hice::rand_uniform({4, 7, 4, 6}, 1.0, 10.0, options);
  Tensor transposed_tensor = hice::transpose(tensor);
  EXPECT_EQ(transposed_tensor.ndim(), 4);
  EXPECT_EQ(transposed_tensor.size(), tensor.size());
  EXPECT_THAT(transposed_tensor.dims(), ElementsAre(6, 4, 7, 4));
  EXPECT_THAT(transposed_tensor.layout().minor_to_major(), ElementsAre(0, 1, 2, 3));
  TENSOR_EXPECT_EQ(tensor, hice::transpose(transposed_tensor));

  Tensor tensor2 = hice::rand_uniform({2, 3, 4}, 1.0, 10.0, options);
  Tensor transposed_tensor2 = hice::transpose(tensor2, {2, 0, 1});
  EXPECT_EQ(transposed_tensor2.ndim(), 3);
  EXPECT_EQ(transposed_tensor2.size(), tensor2.size());
  EXPECT_THAT(transposed_tensor2.dims(), ElementsAre(4, 2, 3));
  EXPECT_THAT(transposed_tensor2.layout().minor_to_major(), ElementsAre(0, 2, 1));
  TENSOR_EXPECT_EQ(tensor2, hice::transpose(transposed_tensor2, {1, 2, 0}));
}

void TransposeMatrixTestDevice(DeviceType device_type) {

  TensorOptions options = dtype(kFloat).device(device_type);
  
  Tensor scalar = hice::rand_uniform({}, 1.0, 10.0, options);
  Tensor transposed_scalar = hice::transpose_matrix(scalar);
  EXPECT_EQ(transposed_scalar.ndim(), 0);
  EXPECT_EQ(transposed_scalar.size(), scalar.size());
  EXPECT_THAT(transposed_scalar.dims(), IsEmpty());
  EXPECT_THAT(transposed_scalar.layout().minor_to_major(), IsEmpty());
  TENSOR_EXPECT_EQ(scalar, hice::transpose_matrix(transposed_scalar));

  Tensor vector = hice::rand_uniform({2}, 1.0, 10.0, options);
  Tensor transposed_vector = hice::transpose_matrix(vector);
  EXPECT_EQ(transposed_vector.ndim(), 1);
  EXPECT_EQ(transposed_vector.size(), vector.size());
  EXPECT_THAT(transposed_vector.dims(), ElementsAre(2));
  EXPECT_THAT(transposed_vector.layout().minor_to_major(), ElementsAre(0));
  TENSOR_EXPECT_EQ(vector, hice::transpose_matrix(transposed_vector));

  Tensor matrix = hice::rand_uniform({3, 4}, 1.0, 10.0, options);
  Tensor transposed_matrix = hice::transpose_matrix(matrix);
  EXPECT_EQ(transposed_matrix.ndim(), 2);
  EXPECT_EQ(transposed_matrix.size(), matrix.size());
  EXPECT_THAT(transposed_matrix.dims(), ElementsAre(4, 3));
  EXPECT_THAT(transposed_matrix.layout().minor_to_major(), ElementsAre(0, 1));
  TENSOR_EXPECT_EQ(matrix, hice::transpose_matrix(transposed_matrix));

  Tensor tensor = hice::rand_uniform({4, 7, 4, 6}, 1.0, 10.0, options);
  Tensor transposed_tensor = hice::transpose_matrix(tensor);
  EXPECT_EQ(transposed_tensor.ndim(), 4);
  EXPECT_EQ(transposed_tensor.size(), tensor.size());
  EXPECT_THAT(transposed_tensor.dims(), ElementsAre(4, 7, 6, 4));
  EXPECT_THAT(transposed_tensor.layout().minor_to_major(), ElementsAre(3, 2, 0, 1));
  TENSOR_EXPECT_EQ(tensor, hice::transpose_matrix(transposed_tensor));
}

TEST(TransposeTest, Transpose) {
  TransposeTestDevice(kCPU);
  TransposeTestDevice(kCUDA);
}

TEST(TransposeTest, transpose_matrix) {
  TransposeMatrixTestDevice(kCPU);
  TransposeMatrixTestDevice(kCUDA);
}

}  // namespace
}  // namespace hice