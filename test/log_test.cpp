#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/new_ops7/log.h"
}

void log_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 0;
  for (int i = 0; i < size; ++i) {
    value += 1;
    data[i] = value;
  }
}

void log_assign_float(Tensor t) {
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

TEST(Log, Float) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  log_assign_float(input);

  Tensor output;
  aitisa_log(input, &output);

  auto* out_data = (double*)aitisa_tensor_data(output);
  double test_data[] = {-0.693147, 0.000000, 0.405465,
                        0.693147,  0.916291, 1.098612};
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

TEST(Log, Int32) {
  Tensor input;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  log_assign_int32(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_log(input, &output);
  // tensor_printer2d(output);

  auto* out_data = (double*)aitisa_tensor_data(output);
  double test_data[] = {0.000000, 0.693147, 1.098612,
                        1.386294, 1.609438, 1.791759};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api