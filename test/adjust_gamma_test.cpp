#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops3/adjust_gamma.h"
//#include "src/tool/tool.h"
}

void adjust_gamma_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i + 1.0;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(AdjustBrightness, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  adjust_gamma_assign_float(input);
  // tensor_printer2d(input);
  double gain = 1.0;
  double gamma = 0.2;

  Tensor output;
  aitisa_adjust_gamma(input, gain, gamma, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {1., 1.14869835, 1.24573094, 1.31950791, 1.37972966, 1.43096908,
                       1.47577316, 1.51571657, 1.55184557, 1.58489319, 1.61539427, 1.64375183,
                       1.67027765, 1.6952182, 1.71877193, 1.74110113, 1.76234035, 1.78260246};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(AdjustBrightness, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  adjust_gamma_assign_float(input);
  // tensor_printer2d(input);
  double gain = 2.0;
  double gamma = 0.3;

  Tensor output;
  aitisa_adjust_gamma(input, gain, gamma, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {2., 2.46228883, 2.78077834, 3.03143313, 3.24131319, 3.42353972,
                       3.58557993, 3.73213197, 3.86636409, 3.99052463, 4.10627283, 4.2148718,
                       4.31730769, 4.41436669, 4.50668676, 4.59479342, 4.67912527, 4.76005255,
                       4.83789096, 4.9129121, 4.9853515, 5.05541485, 5.123283, 5.18911587,
                       5.25305561, 5.31522924, 5.37575076, 5.43472289, 5.49223859, 5.54838223,
                       5.6032307, 5.65685425, 5.70931727, 5.76067893, 5.81099376, 5.8603121};
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