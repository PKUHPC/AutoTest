#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops5/image_gradients.h"
//#include "src/tool/tool.h"
}

void image_gradients_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
      value = (float)i * 0.1;
      data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(ImageGradients, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  image_gradients_assign_float(input);
  // tensor_printer2d(input);

  Tensor grad_x;
  Tensor grad_y;
  aitisa_image_gradients(input, &grad_x, &grad_y);
  // tensor_printer2d(output);

  float* out_data1 = (float*)aitisa_tensor_data(grad_x);
  float* out_data2 = (float*)aitisa_tensor_data(grad_y);
  float test_data1[] = {0.3, 0.3, 0.3, 0.0, 0.0, 0.0,
                        0.3, 0.3, 0.3, 0.0, 0.0, 0.0,
                        0.3, 0.3, 0.3, 0.0, 0.0, 0.0};
  float test_data2[] = {0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
                        0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
                        0.1, 0.1, 0.0, 0.1, 0.1, 0.0};
  int64_t size = aitisa_tensor_size(grad_x);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data1[i] - test_data1[i]) < 0.000001);
    EXPECT_TRUE(abs(out_data2[i] - test_data2[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&grad_x);
  aitisa_destroy(&grad_y);
}

TEST(ImageGradients, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  image_gradients_assign_float(input);
  // tensor_printer2d(input);

  Tensor grad_x;
  Tensor grad_y;
  aitisa_image_gradients(input, &grad_x, &grad_y);
  // tensor_printer2d(output);

  float* out_data1 = (float*)aitisa_tensor_data(grad_x);
  float* out_data2 = (float*)aitisa_tensor_data(grad_y);
  float test_data1[] = {0.3, 0.3, 0.3, 0.0, 0.0, 0.0,
                        0.3, 0.3, 0.3, 0.0, 0.0, 0.0,
                        0.3, 0.3, 0.3, 0.0, 0.0, 0.0,
                        0.3, 0.3, 0.3, 0.0, 0.0, 0.0,
                        0.3, 0.3, 0.3, 0.0, 0.0, 0.0,
                        0.3, 0.3, 0.3, 0.0, 0.0, 0.0};
  float test_data2[] = {0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
                        0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
                        0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
                        0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
                        0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
                        0.1, 0.1, 0.0, 0.1, 0.1, 0.0};
  int64_t size = aitisa_tensor_size(grad_x);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data1[i] - test_data1[i]) < 0.000001);
    EXPECT_TRUE(abs(out_data2[i] - test_data2[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&grad_x);
  aitisa_destroy(&grad_y);
}
}  // namespace
}  // namespace aitisa_api