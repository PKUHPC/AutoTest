#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops2/rgb_to_grayscale.h"
//#include "src/tool/tool.h"
}

void rgb_to_grayscale_assign_float(Tensor t) {
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

TEST(Rgb2GrayScale, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  rgb_to_grayscale_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_rgb_to_grayscale(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.189, 0.289, 0.389, 0.489, 0.589, 0.689};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Rgb2GrayScale, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  rgb_to_grayscale_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_rgb_to_grayscale(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.189, 0.289, 0.389, 0.489, 0.589, 0.689,
                       1.989, 2.089, 2.189, 2.289, 2.389, 2.489};
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