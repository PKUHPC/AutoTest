#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/new_ops8/compare.h"
//#include "src/tool/tool.h"
}

void compare_natural_assign1(Tensor t) {
  int64_t ndim = aitisa_tensor_ndim(t);
  int64_t* dims = aitisa_tensor_dims(t);
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1;
    data[i] = value;
  }
}

void compare_natural_assign2(Tensor t) {
  int64_t ndim = aitisa_tensor_ndim(t);
  int64_t* dims = aitisa_tensor_dims(t);
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.2;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Binary, equal) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[1] = {4};
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor2);
  compare_natural_assign1(tensor1);
  compare_natural_assign1(tensor2);
  Tensor output;
  aitisa_equal(tensor1, tensor2, &output);
  /*
  tensor_printer(tensor1);
  tensor_printer(tensor2);
  tensor_printer(output);
  */
  int64_t expected_ndim = 1;
  int64_t expected_dims[1] = {4};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  auto* data = (int32_t*)aitisa_tensor_data(output);
  float result[4] = {1, 1, 1, 1};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Binary, greater) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[1] = {4};
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor2);
  compare_natural_assign2(tensor1);
  compare_natural_assign1(tensor2);
  Tensor output;
  aitisa_greater(tensor1, tensor2, &output);
  /*
  tensor_printer(tensor1);
  tensor_printer(tensor2);
  tensor_printer(output);
  */
  int64_t expected_ndim = 1;
  int64_t expected_dims[1] = {4};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  auto* data = (int32_t*)aitisa_tensor_data(output);
  float result[4] = {0, 1, 1, 1};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Binary, greater_equal) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[1] = {4};
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor2);
  compare_natural_assign1(tensor1);
  compare_natural_assign1(tensor2);
  Tensor output;
  aitisa_greater_equal(tensor1, tensor2, &output);
  /*
  tensor_printer(tensor1);
  tensor_printer(tensor2);
  tensor_printer(output);
  */
  int64_t expected_ndim = 1;
  int64_t expected_dims[1] = {4};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  int32_t* data = (int32_t*)aitisa_tensor_data(output);
  float result[4] = {
      1,
      1,
      1,
      1,
  };
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}

TEST(Binary, less) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[1] = {4};
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor2);
  compare_natural_assign1(tensor1);
  compare_natural_assign1(tensor2);
  Tensor output;
  aitisa_less(tensor1, tensor2, &output);
  /*
  tensor_printer(tensor1);
  tensor_printer(tensor2);
  tensor_printer(output);
  */
  int64_t expected_ndim = 1;
  int64_t expected_dims[1] = {4};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  auto* data = (int32_t*)aitisa_tensor_data(output);
  float result[4] = {0, 0, 0, 0};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
  std::cout << std::endl;
}

TEST(Binary, less_equal) {
  Tensor tensor1;
  Tensor tensor2;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[1] = {4};
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor1);
  aitisa_create(dtype, device, dims, 1, NULL, 0, &tensor2);
  compare_natural_assign1(tensor1);
  compare_natural_assign1(tensor2);
  Tensor output;
  aitisa_less_equal(tensor1, tensor2, &output);
  /*
  tensor_printer(tensor1);
  tensor_printer(tensor2);
  tensor_printer(output);
  */
  int64_t expected_ndim = 1;
  int64_t expected_dims[1] = {4};
  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
  for (int i = 0; i < expected_ndim; ++i) {
    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
  }
  auto* data = (int32_t*)aitisa_tensor_data(output);
  float result[4] = {1, 1, 1, 1};
  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
    EXPECT_FLOAT_EQ(result[i], data[i]);
  }
  aitisa_destroy(&tensor1);
  aitisa_destroy(&tensor2);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api