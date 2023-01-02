#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/new_ops8/arg_reduce.h"
}

void min_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  auto* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 0;
  for (int i = 0; i < size; ++i) {
    value += 1;
    data[i] = value;
  }
}

void min_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  auto* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value += 1;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Min, Float) {
  Tensor input, output;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t input_dims[4] = {2, 2, 2, 3};

  aitisa_create(dtype, device, input_dims, 4, nullptr, 0, &input);
  min_assign_float(input);

  int64_t dim = 3;
  aitisa_min(input, dim, 0, &output);

  float test_data[] = {1, 3, 8, 10, 13, 16, 19, 22};

  int64_t output_size = aitisa_tensor_size(output);
  auto* out_data = (float*)aitisa_tensor_data(output);

  for (int64_t i = 0; i < output_size; i++) {
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Min, Int32) {
  Tensor input, output;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t input_dims[4] = {2, 2, 2, 2};

  aitisa_create(dtype, device, input_dims, 4, nullptr, 0, &input);
  min_assign_int32(input);

  int64_t dim = 2;
  aitisa_min(input, dim, 0, &output);

  int32_t test_data[] = {1, 2, 5, 6, 9, 10, 13, 14};

  int64_t output_size = aitisa_tensor_size(output);
  auto* out_data = (int32_t*)aitisa_tensor_data(output);

  for (int64_t i = 0; i < output_size; i++) {
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api