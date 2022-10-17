#include "gtest/gtest.h"
extern "C" {
#include "src/new_ops3/random_contrast.h"
//#include "src/tool/tool.h"
}

void random_contrast_assign_float(Tensor t) {
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
TEST(RandomContrast, Float4d) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[4] = {2, 3, 2, 3};
  aitisa_create(dtype, device, dims, 4, NULL, 0, &input);
  random_contrast_assign_float(input);
  // tensor_printer2d(input);

  Tensor output;
  aitisa_random_contrast(input, 0.3, 0.5, &output);
  // tensor_printer2d(output);
  srand(0);
  double factor = (rand() / double(RAND_MAX)) * (0.5 - 0.3) + 0.3;

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0., 0.00827619, 0.04905503, 0.08983386, 0.13061269, 0.17139153,
                       0.13416403, 0.17494286, 0.21572169, 0.25650053, 0.29727936, 0.33805819,
                       0.30083069, 0.34160953, 0.38238836, 0.42316719, 0.46394603, 0.50472486,
                       0.46749736, 0.50827619, 0.54905503, 0.58983386, 0.63061269, 0.67139153,
                       0.63416403, 0.67494286, 0.71572169, 0.75650053, 0.79727936, 0.83805819,
                       0.80083069, 0.84160953, 0.88238836, 0.92316719, 0.96394603, 1.};
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