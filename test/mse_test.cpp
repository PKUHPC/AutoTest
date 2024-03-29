#include "gtest/gtest.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/new_ops6/mse_loss.h"
}

void mse_loss_assign_int32(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = -5;
  for (int i = 0; i < size; ++i) {
    value += 1;
    data[i] = value;
  }
}

void mse_loss_assign_int32_target(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  int32_t* data = (int32_t*)aitisa_tensor_data(t);
  int32_t value = 5;
  for (int i = 0; i < size; ++i) {
    value += 0.5;
    data[i] = value;
  }
}

void mse_loss_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = -2;
  for (int i = 0; i < size; ++i) {
    value += 0.5;
    data[i] = value;
  }
}

void mse_loss_assign_float_target(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 2;
  for (int i = 0; i < size; ++i) {
    value += 0.1;
    data[i] = value;
  }
}

void mse_loss_assign_float_weight(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 1;
  for (int i = 0; i < size; ++i) {
    value += 1;
    data[i] = value;
  }
}
namespace aitisa_api {
namespace {

TEST(MseLoss, Float) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  mse_loss_assign_float(input);

  Tensor target;
  aitisa_create(dtype, device, dims, 2, NULL, 0, &target);
  mse_loss_assign_float_target(target);

  Tensor weight;
  aitisa_create(dtype, device, dims, 2, NULL, 0, &weight);
  mse_loss_assign_float_weight(weight);

  Tensor output;
  aitisa_mse_loss(input, target, weight, 0, &output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {25.919998, 30.719997, 31.359993,
                       28.799992, 23.999989, 17.919989};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.00001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.00001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&target);
  aitisa_destroy(&weight);
  aitisa_destroy(&output);
}

TEST(MseLoss, Int32) {
  Tensor input;
  DataType dtype = kInt32;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {2, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  mse_loss_assign_int32(input);

  Tensor target;
  aitisa_create(dtype, device, dims, 2, NULL, 0, &target);
  mse_loss_assign_int32_target(target);

  Tensor output;
  aitisa_mse_loss(input, target, {}, 0, &output);

  int32_t* out_data = (int32_t*)aitisa_tensor_data(output);
  int32_t test_data[] = {81, 64, 49, 36, 25, 16};
  int64_t size = aitisa_tensor_size(input);
  for (int64_t i = 0; i < size; i++) {
    EXPECT_EQ(out_data[i], test_data[i]);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&target);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api