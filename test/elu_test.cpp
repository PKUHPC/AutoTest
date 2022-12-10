#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/new_ops7/elu.h"
}

void exp_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = -5;
  for (int i = 0; i < size; ++i) {
    value += 1;
    data[i] = value;
  }
}

void exp_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = -2;
  for (int i = 0; i < size; ++i) {
    value += 0.5;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Elu, Float) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  exp_assign_float(input);
  float alpha = 1.0;
  Tensor output;
  aitisa_elu(input, alpha, &output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {-0.776870, -0.632121, -0.393469,
                       0.000000,  0.500000,  1.000000};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }
  std::cout << std::endl;
  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Elu, Int32) {
  Tensor input;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  exp_assign_int32(input);
  // tensor_printer2d(input);

  Tensor output;
  float alpha = 1.0;
  aitisa_elu(input, alpha, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {-0.981684, -0.950213, -0.864665,
                       -0.632121, 0.000000,  1.000000};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api