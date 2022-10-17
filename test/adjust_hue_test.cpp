#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops3/adjust_hue.h"
//#include "src/tool/tool.h"
}

void adjust_hue_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = (float)i;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(AdjustHue, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  adjust_hue_assign_float(input);
  // tensor_printer2d(input);
  double adjust_factor = 50;

  Tensor output;
  aitisa_adjust_hue(input, adjust_factor, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.0001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.0001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(AdjustHue, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  adjust_hue_assign_float(input);
  // tensor_printer2d(input);
  double adjust_factor = 1.5;

  Tensor output;
  aitisa_adjust_hue(input, adjust_factor, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0, 1, 2, 3, 4, 5,
                       12, 13, 14, 15, 16, 17,
                       12, 13, 14, 15, 16, 17,
                       18, 19, 20, 21, 22, 23,
                       30, 31, 32, 33, 34, 35,
                       30, 31, 32, 33, 34, 35};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.0001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.0001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api