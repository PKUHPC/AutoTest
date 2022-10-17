#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops3/adjust_brightness.h"
//#include "src/tool/tool.h"
}

void adjust_brightness_assign_float(Tensor t) {
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

TEST(AdjustBrightness, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  adjust_brightness_assign_float(input);
  // tensor_printer2d(input);
  double adjust_factor = 0.1;

  Tensor output;
  aitisa_adjust_brightness(input, adjust_factor, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.1, 0.15555556, 0.21111111, 0.26666667, 0.32222222, 0.37777778,
                       0.43333333, 0.48888889, 0.54444444, 0.6, 0.65555556, 0.71111111,
                       0.76666667, 0.82222222, 0.87777778, 0.93333333, 0.98888889, 1.0};
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
  adjust_brightness_assign_float(input);
  // tensor_printer2d(input);
  double adjust_factor = -0.1;

  Tensor output;
  aitisa_adjust_brightness(input, adjust_factor, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0., 0., 0., 0., 0.01111111, 0.03888889,
                       0.06666667, 0.09444444, 0.12222222, 0.15, 0.17777778, 0.20555556,
                       0.23333333, 0.26111111, 0.28888889, 0.31666667, 0.34444444, 0.37222222,
                       0.4, 0.42777778, 0.45555556, 0.48333333, 0.51111111, 0.53888889,
                       0.56666667, 0.59444444, 0.62222222, 0.65, 0.67777778, 0.70555556,
                       0.73333333, 0.76111111, 0.78888889, 0.81666667, 0.84444444, 0.87222222};
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