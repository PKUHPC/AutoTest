#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops2/rgb_to_yuv.h"
//#include "src/tool/tool.h"
}

void rgb_to_yuv_assign_float(Tensor t) {
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

TEST(Rgb2Yuv, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  rgb_to_yuv_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_rgb_to_yuv(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.186, 0.286, 0.386, 0.486, 0.586, 0.686,
                       0.352002, 0.352002, 0.352002, 0.352002, 0.352002, 0.352002,
                       -0.426222, -0.426222, -0.426222, -0.426222, -0.426222, -0.426222};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Rgb2Yuv, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  rgb_to_yuv_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_rgb_to_yuv(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.186, 0.286, 0.386, 0.486, 0.586, 0.686,
                       0.352002, 0.352002, 0.352002, 0.352002, 0.352002, 0.352002,
                       -0.426222, -0.426222, -0.426222, -0.426222, -0.426222, -0.426222,
                       1.986, 2.086, 2.186, 2.286, 2.386, 2.486,
                       0.352002, 0.352002, 0.352002, 0.352002, 0.352002, 0.352002,
                       -0.426222, -0.426222, -0.426222, -0.426222, -0.426222, -0.426222};
  int64_t size = aitisa_tensor_size(input);
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