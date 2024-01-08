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
      0.523152, 0.286846, 0.518161, 0.575174, 0.568285, 0.473986, 0.523936,
      0.561420, 0.518561, 0.296706, 0.531083, 0.568210, 0.567156, 0.487563,
      0.521355, 0.550063, 0.526049, 0.304177, 0.516874, 0.580389, 0.577912,
      0.468929, 0.510792, 0.562494, 0.508793, 0.300885, 0.536643, 0.561208,
      0.568428, 0.485775, 0.505172, 0.551363, 0.517031, 0.305166, 0.534881,
      0.566407, 0.580072, 0.493924, 0.526406, 0.540590, 0.514796, 0.298756,
      0.524845, 0.579555, 0.570881, 0.480768, 0.520503, 0.553405, 0.527170,
      0.297921, 0.520971, 0.577582, 0.574129, 0.479505, 0.524804, 0.557708,
      0.515964, 0.298526, 0.517448, 0.573973, 0.564367, 0.471809, 0.500992,
      0.566778,
  };
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