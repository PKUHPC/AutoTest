#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops2/yuv_to_rgb.h"
//#include "src/tool/tool.h"
}

void yuv_to_rgb_assign_float(Tensor t) {
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

TEST(Yuv2Rgb, Float3d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[3] = {3, 2, 3};
  aitisa_create(dtype, device, dims, 3, NULL, 0, &input);
  yuv_to_rgb_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_yuv_to_rgb(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.96675, 1.2075, 1.44825, 1.689, 1.92975, 2.1705,
                       -1.04886, -1.0551, -1.06134, -1.06758, -1.07382, -1.08006,
                       0.2337, 0.5116, 0.7895, 1.0674, 1.3453, 1.6232};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Yuv2Rgb, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  yuv_to_rgb_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_yuv_to_rgb(input, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.96675, 1.2075, 1.44825, 1.689, 1.92975, 2.1705,
                       -1.04886, -1.0551, -1.06134, -1.06758, -1.07382, -1.08006,
                       0.2337, 0.5116, 0.7895, 1.0674, 1.3453, 1.6232,
                       5.30025, 5.541, 5.78175, 6.0225, 6.26325, 6.504,
                       -1.16118, -1.16742, -1.17366, -1.1799, -1.18614, -1.19238,
                       5.2359, 5.5138, 5.7917, 6.0696, 6.3475, 6.6254};
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