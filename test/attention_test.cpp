#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

extern "C" {
#include "src/basic/factories.h"
#include "src/nn/attention.h"
//#include "src/tool/tool.h"
}

namespace aitisa_api {
namespace {

TEST(Attention, test1) {
  Tensor query, key, value;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t query_dims[4] = {1, 32, 1, 32};
  int64_t key_dims[4] = {1, 32, 1, 32};
  int64_t value_dims[4] = {1, 32, 1, 32};

  Tensor output;


  aitisa_full(dtype, device, query_dims, 4, 2.0, &query);
  aitisa_full(dtype, device, key_dims, 4, 2.0, &key);
  aitisa_full(dtype, device, value_dims, 4, 1.0, &value);

  int is_causal = 0;

  aitisa_attention(query, key, value, is_causal, &output);
  /*
  tensor_printer2d(input);
  tensor_printer2d(filter);
  tensor_printer2d(output);
  */
   int64_t *output_dims = aitisa_tensor_dims(output);
//   for (int i = 0; i < 4; ++i) {
//    std::cout << output_dims[i] << ", ";
//  }
//   std::cout << std::endl;

  float *output_data = (float *)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  for (int i = 0; i < output_size; ++i) {
//     std::cout << output_data[i] << ", ";
    EXPECT_EQ(1, output_data[i]);
  }
  // std::cout << std::endl;

  aitisa_destroy(&query);
  aitisa_destroy(&key);
  aitisa_destroy(&value);

  aitisa_destroy(&output);
}

}  // namespace
}  // namespace aitisa_api