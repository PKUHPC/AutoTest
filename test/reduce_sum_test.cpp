#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/new_ops8/reduce_sum.h"
}

void reduce_sum_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 0;
  for (int i = 0; i < size; ++i) {
    value += 1;
    data[i] = value;
  }
}

void reduce_sum_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value += 0.5;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Reduce_sum, Float) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t input_dims[4] = {2, 2, 3, 3};
  aitisa_create(dtype, device, input_dims, 4, NULL, 0, &input);
  reduce_sum_assign_float(input);
  int64_t dims[2] = {2, 1};
  int64_t dims_length = 2;
  Tensor output;
  aitisa_reduce_sum(input, dims, dims_length, 1, &output);

  //  auto* out_data = (double*)aitisa_tensor_data(output);
  double test_data[] = {-0.693147, 0.000000, 0.405465,
                        0.693147,  0.916291, 1.098612};
  int64_t size = aitisa_tensor_size(input);
  int64_t* out_dims = aitisa_tensor_dims(output);
  int64_t out_ndim = aitisa_tensor_ndim(output);
  for (int64_t i = 0; i < out_ndim; i++) {
    std::cout << out_dims[i] << " , ";
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    //    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }
  std::cout << std::endl;
  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Reduce_sum, Int32) {
  Tensor input;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t input_dims[4] = {2, 2, 3, 3};
  aitisa_create(dtype, device, input_dims, 4, NULL, 0, &input);
  reduce_sum_assign_int32(input);
  int64_t dims[2] = {2, 1};
  int64_t dims_length = 2;
  Tensor output;
  aitisa_reduce_sum(input, dims, dims_length, 0, &output);
  // tensor_printer2d(output);
  int64_t* out_dims = aitisa_tensor_dims(output);
  int64_t out_ndim = aitisa_tensor_ndim(output);
  for (int64_t i = 0; i < out_ndim; i++) {
    std::cout << out_dims[i] << " , ";
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    //    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }
  //  auto* out_data = (double*)aitisa_tensor_data(output);
  double test_data[] = {0.000000, 0.693147, 1.098612,
                        1.386294, 1.609438, 1.791759};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    //    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api