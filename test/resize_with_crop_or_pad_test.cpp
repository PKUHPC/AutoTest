#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops4/resize_with_crop_or_pad.h"
//#include "src/tool/tool.h"
}

void resize_with_crop_or_pad_assign_float(Tensor t) {
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

TEST(ResizeWithCropOrPad, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 5, 2};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  resize_with_crop_or_pad_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_resize_with_crop_or_pad(input, 3, 4, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.0, 0.2, 0.3, 0.0, 0.0, 0.4, 0.5, 0.0, 0.0, 0.6, 0.7, 0.0,
                       0.0, 1.2, 1.3, 0.0, 0.0, 1.4, 1.5, 0.0, 0.0, 1.6, 1.7, 0.0,
                       0.0, 2.2, 2.3, 0.0, 0.0, 2.4, 2.5, 0.0, 0.0, 2.6, 2.7, 0.0};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(ResizeWithCropOrPad, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 5, 5};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  resize_with_crop_or_pad_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_resize_with_crop_or_pad(input, 2, 2, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.6, 0.7, 1.1, 1.2,
                       3.1, 3.2, 3.6, 3.7,
                       5.6, 5.7, 6.1, 6.2,
                       8.1, 8.2, 8.6, 8.7,
                       10.6, 10.7, 11.1, 11.2,
                       13.1, 13.2, 13.6, 13.7};
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