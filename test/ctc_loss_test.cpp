#include "gtest/gtest.h"
extern "C" {
#include <inttypes.h>

#include "src/basic/factories.h"
#include "src/new_ops6/ctc_loss.h"
#include "src/nn/softmax.h"
}

void ctc_loss_assign_float(Tensor t) {
  int32_t size = aitisa_tensor_size(t);
  auto* data = (float*)aitisa_tensor_data(t);
  float init = 0;
  float max = 1;
  float value = init;
  for (int i = 0; i < size; ++i) {
    value += (max - init) / size;
    data[i] = value;
  }
}

void ctc_loss_assign_target(Tensor t, int classes) {
  int32_t size = aitisa_tensor_size(t);
  auto* data = (int32_t*)aitisa_tensor_data(t);
  for (int i = 0; i < size; ++i) {
    data[i] = (i % classes) + 1;
  }
}

namespace aitisa_api {
namespace {

TEST(CtcLoss, Float2d) {
  Tensor prod;
  DataType prods_dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t prods_dims[3] = {5, 3, 5};
  aitisa_create(prods_dtype, device, prods_dims, 3, NULL, 0, &prod);
  ctc_loss_assign_float(prod);
  Tensor input;
  aitisa_softmax(prod, 2, &input);
  Tensor target;
  DataType target_dtype = kInt64;
  int64_t target_dims[2] = {3, 4};
  aitisa_full(target_dtype, device, target_dims, 2, 4, &target);
  aitisa_create(target_dtype, device, target_dims, 2, NULL, 0, &target);
  ctc_loss_assign_target(target, 5);

  Tensor probs_lengths, target_lengths;
  int64_t probs_lengths_dims[1] = {3};
  aitisa_full(target_dtype, device, probs_lengths_dims, 1, 5, &probs_lengths);
  aitisa_full(target_dtype, device, probs_lengths_dims, 1, 4, &target_lengths);
  Tensor output;
  int32_t* out_data1 = (int32_t *)aitisa_tensor_data(target);
  std::cout << out_data1[0] << std::endl;
  aitisa_ctc_loss(input, target, probs_lengths, target_lengths, &output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {5.83585, 5.8952, 5.88033};
  int64_t size = aitisa_tensor_size(output);
  for (int32_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
//        std::cout << out_data[i] << std::endl;
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.0001);
  }

  aitisa_destroy(&prod);
  aitisa_destroy(&input);
  aitisa_destroy(&probs_lengths);
  aitisa_destroy(&target_lengths);
  aitisa_destroy(&target);
  aitisa_destroy(&output);
}

}  // namespace
}  // namespace aitisa_api