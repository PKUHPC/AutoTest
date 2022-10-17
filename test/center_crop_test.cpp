#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops4/center_crop.h"
//#include "src/tool/tool.h"
}

void center_crop_assign_float(Tensor t) {
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

TEST(CenterCrop, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 5, 5};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  center_crop_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_center_crop(input, 3, 3, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.6, 1.7, 1.8,
                       3.1, 3.2, 3.3, 3.6, 3.7, 3.8, 4.1, 4.2, 4.3,
                       5.6, 5.7, 5.8, 6.1, 6.2, 6.3, 6.6, 6.7, 6.8};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(CenterCrop, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 5, 5};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  center_crop_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_center_crop(input, 2, 2, &output);
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