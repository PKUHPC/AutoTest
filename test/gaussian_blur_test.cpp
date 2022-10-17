#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops5/gaussian_blur.h"
//#include "src/tool/tool.h"
}

void gaussian_blur_assign_float(Tensor t) {
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

TEST(GaussianBlur, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 3, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  gaussian_blur_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_gaussian_blur(input, 3, 1.0, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.088888889, 0.166666667, 0.133333333,
                       0.233333333, 0.4, 0.3,
                       0.222222222, 0.366666667, 0.266666667,
                       0.488888889, 0.766666667, 0.533333333,
                       0.833333333, 1.3, 0.9,
                       0.622222222, 0.966666667, 0.666666667,
                       0.888888889, 1.366666667, 0.933333333,
                       1.433333333, 2.2, 1.5,
                       1.022222222, 1.566666667, 1.066666667};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(GaussianBlur, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 3, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  gaussian_blur_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_gaussian_blur(input, 3, 1.0, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.088889, 0.166667, 0.133333, 0.233333, 0.400000, 0.300000, 0.222222, 0.366667, 0.266667,
                       0.488889, 0.766667, 0.533333, 0.833333, 1.300000, 0.900000, 0.622222, 0.966667, 0.666667,
                       0.888889, 1.366667, 0.933333, 1.433333, 2.200000, 1.500000, 1.022222, 1.566667, 1.066667,
                       1.288889, 1.966667, 1.333333, 2.033333, 3.100000, 2.100000, 1.422222, 2.166667, 1.466667,
                       1.688889, 2.566667, 1.733333, 2.633333, 4.000000, 2.700000, 1.822222, 2.766667, 1.866667,
                       2.088889, 3.166667, 2.133333, 3.233333, 4.900000, 3.300000, 2.222222, 3.366667, 2.266667};
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