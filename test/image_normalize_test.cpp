#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops3/image_normalize.h"
//#include "src/tool/tool.h"
}

void image_normalize_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = (float)i / size;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(ImageNormalize, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  image_normalize_assign_float(input);
  // tensor_printer2d(input);
  double mean[3] = {0, 0, 0};
  double std[3] = {1.0, 1.0, 1.0};

  Tensor output;
  aitisa_image_normalize(input, mean, std, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.0, 0.0555556, 0.1111111, 0.1666667, 0.2222222, 0.2777778,
                       0.3333333, 0.3888889, 0.4444444, 0.5, 0.5555556, 0.6111111,
                       0.6666667, 0.7222222, 0.7777778, 0.8333333, 0.8888889, 0.9444444};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(ImageNormalize, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  image_normalize_assign_float(input);
  // tensor_printer2d(input);
  double mean[3] = {0, 0, 0};
  double std[3] = {0.5, 0.5, 0.5};

  Tensor output;
  aitisa_image_normalize(input, mean, std, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.0, 0.0555556, 0.1111111, 0.1666667, 0.2222222, 0.2777778,
                       0.3333333, 0.3888889, 0.4444444, 0.5, 0.5555556, 0.6111111,
                       0.6666667, 0.7222222, 0.7777778, 0.8333333, 0.8888889, 0.9444444,
                       1.0, 1.0555556, 1.1111111, 1.1666667, 1.2222222, 1.2777778,
                       1.3333333, 1.3888889, 1.4444444, 1.5, 1.5555556, 1.6111111,
                       1.6666667, 1.7222222, 1.7777778, 1.8333333, 1.8888889, 1.9444444};
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