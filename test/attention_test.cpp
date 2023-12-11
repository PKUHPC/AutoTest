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
  int64_t query_dims[4] = {1, 8, 1, 8};
  int64_t key_dims[4] = {1, 8, 1, 8};
  int64_t value_dims[4] = {1, 8, 1, 8};
  double a, b;
  long int s;
  double test_data[] = {
      0.623062, 0.162065, 0.42201,  0.511313, 0.635983, 0.584241, 0.772669,
      0.607776, 0.515369, 0.305455, 0.530901, 0.567454, 0.569122, 0.480134,
      0.499052, 0.561439, 0.515369, 0.305455, 0.530901, 0.567454, 0.569122,
      0.480134, 0.499052, 0.561439, 0.515369, 0.305455, 0.530901, 0.567454,
      0.569122, 0.480134, 0.499052, 0.561439, 0.515369, 0.305455, 0.530901,
      0.567454, 0.569122, 0.480134, 0.499052, 0.561439, 0.515369, 0.305455,
      0.530901, 0.567454, 0.569122, 0.480134, 0.499052, 0.561439, 0.515369,
      0.305455, 0.530901, 0.567454, 0.569122, 0.480134, 0.499052, 0.561439,
      0.515369, 0.305455, 0.530901, 0.567454, 0.569122, 0.480134, 0.499052,
      0.561439};
  a = 0.0;
  b = 1.0;
  s = 13579;

  Tensor output;

  aitisa_uniform(dtype, device, query_dims, 4, a, b,&s, &query);
  aitisa_uniform(dtype, device, key_dims, 4, a, b,&s, &key);
  aitisa_uniform(dtype, device, value_dims, 4, a, b,&s, &value);

  int is_causal = 0;

  aitisa_attention(query, key, value, is_causal, &output);
  int64_t *output_dims = aitisa_tensor_dims(output);

  float *output_data = (float *)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  for (int i = 0; i < output_size; ++i) {
     EXPECT_TRUE(abs(output_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&query);
  aitisa_destroy(&key);
  aitisa_destroy(&value);

  aitisa_destroy(&output);
}

}  // namespace
}  // namespace aitisa_api