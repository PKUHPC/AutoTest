#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops6/cross_entropy.h"
//#include "src/tool/tool.h"
}

void cross_entropy_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float init = -2;
  float max = 2;
  float value = init;
  for (int i = 0; i < size; ++i) {
    value += (max - init) / size;
    data[i] = value;
  }
}

void cross_entropy_assign_target(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  for (int i = 0; i < size; ++i) {
    data[i] = (i == 0 || i == 6 || i == 12) ? 1 : 0;
  }
}

namespace aitisa_api {
namespace {

TEST(CrossEntropyLoss, Float2d) {
  Tensor prod;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {3, 5};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &prod);
  cross_entropy_assign_float(prod);

  Tensor target;
  aitisa_create(kFloat, device, dims, 2, NULL, 0, &target);
  cross_entropy_assign_target(target);

  Tensor output;
  aitisa_cross_entropy(prod, target, NULL, &output);
  // tensor_printer2d(output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {2.21282, 1.94615, 1.67948};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.0001);
  }

  aitisa_destroy(&prod);
  aitisa_destroy(&target);
  aitisa_destroy(&output);
}

}  // namespace
}  // namespace aitisa_api