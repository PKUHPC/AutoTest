#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops2/rgb_to_hsv.h"
//#include "src/tool/tool.h"
}

void rgb_to_hsv_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Rgb2Hsv, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  rgb_to_hsv_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_rgb_to_hsv(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {210, 210, 210, 210, 210, 210,
                       1.0, 0.923076, 0.857142, 0.8, 0.75, 0.705882,
                       0.0047059, 0.005098, 0.0054902, 0.0058824, 0.0062745, 0.0066667};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Rgb2Hsv, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  rgb_to_hsv_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_rgb_to_hsv(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {210, 210, 210, 210, 210, 210,
                       1.0, 0.923076, 0.857142, 0.8, 0.75, 0.705882,
                       0.0047059, 0.005098, 0.0054902, 0.0058824, 0.0062745, 0.0066667,
                       210, 210, 210, 210, 210, 210,
                       0.4, 0.3870968, 0.375, 0.3636364, 0.3529412, 0.3428571,
                       0.0117647, 0.0121569, 0.012549, 0.0129412, 0.0133333, 0.0137255};
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