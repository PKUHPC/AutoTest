#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops2/hsv_to_rgb.h"
// #include "src/tool/tool.h"
}

void hsv_to_rgb_assign_float(Tensor t) {
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

TEST(Hsv2Rgb, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};

  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  hsv_to_rgb_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_hsv_to_rgb(input, &output);
  // tensor_printer2d(output);

  float *out_data = (float *)aitisa_tensor_data(output);
  float test_data[] = {306., 331.5, 357., 382.5, 408., 433.5,
                       122.4, 99.45, 71.4, 38.25, 0., -43.35,
                       122.4, 99.45, 71.4, 38.25, 0., -43.35};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.0001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.0001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api