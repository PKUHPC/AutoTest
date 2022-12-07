#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/new_ops7/neg.h"
}

void neg_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = -5;
  for (int i = 0; i < size; ++i) {
    value += 1;
    data[i] = value;
  }
}

void neg_assign_float(Tensor t) {
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

TEST(Neg, Float) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  neg_assign_float(input);

  Tensor output;
  aitisa_neg(input, &output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {1.5, 1, 0.5, -0, -0.5, -1};
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

TEST(Neg, Int32) {
  Tensor input;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  neg_assign_int32(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_neg(input, &output);
  // tensor_printer2d(output);

  int32_t* out_data = (int32_t*)aitisa_tensor_data(output);
  int32_t test_data[] = {4, 3, 2, 1, 0, -1};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    EXPECT_EQ(out_data[i], test_data[i]);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api