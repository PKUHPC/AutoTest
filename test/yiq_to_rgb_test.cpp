#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops2/yiq_to_rgb.h"
//#include "src/tool/tool.h"
}

void yiq_to_rgb_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1 - 0.3;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Yiq2Rgb, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  yiq_to_rgb_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_yiq_to_rgb(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.5448, 0.8024, 1.06, 1.3176, 1.5752, 1.8328,
                       -0.9639, -0.9558, -0.9477, -0.9396, -0.9315, -0.9234,
                       0.9021, 1.0618, 1.2215, 1.3812, 1.5409, 1.7006};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Yiq2Rgb, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  yiq_to_rgb_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_yiq_to_rgb(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.5448, 0.8024, 1.06, 1.3176, 1.5752, 1.8328,
                       -0.9639, -0.9558, -0.9477, -0.9396, -0.9315, -0.9234,
                       0.9021, 1.0618, 1.2215, 1.3812, 1.5409, 1.7006,
                       5.1816, 5.4392, 5.6968, 5.9544, 6.212, 6.4696,
                       -0.8181, -0.81, -0.8019, -0.7938, -0.7857, -0.7776,
                       3.7767, 3.9364, 4.0961, 4.2558, 4.4155, 4.5752};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api