#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops3/adjust_contrast.h"
//#include "src/tool/tool.h"
}

void adjust_contrast_assign_float(Tensor t) {
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

TEST(AdjustContrast, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  adjust_contrast_assign_float(input);
  // tensor_printer2d(input);
  double adjust_factor = 1.5;

  Tensor output;
  aitisa_adjust_contrast(input, adjust_factor, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0., 0., 0.06944444, 0.20833333, 0.34722222, 0.48611111,
                       0.125, 0.26388889, 0.40277778, 0.54166667, 0.68055556, 0.81944444,
                       0.45833333, 0.59722222, 0.73611111, 0.875, 1., 1.};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(AdjustContrast, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  adjust_contrast_assign_float(input);
  // tensor_printer2d(input);
  double adjust_factor = 1.5;

  Tensor output;
  aitisa_adjust_contrast(input, adjust_factor, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0., 0., 0.03472222, 0.10416667, 0.17361111, 0.24305556,
                       0.0625, 0.13194444, 0.20138889, 0.27083333, 0.34027778, 0.40972222,
                       0.22916667, 0.29861111, 0.36805556, 0.4375, 0.50694444, 0.57638889,
                       0.39583333, 0.46527778, 0.53472222, 0.60416667, 0.67361111, 0.74305556,
                       0.5625, 0.63194444, 0.70138889, 0.77083333, 0.84027778, 0.90972222,
                       0.72916667, 0.79861111, 0.86805556, 0.9375, 1., 1.};
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