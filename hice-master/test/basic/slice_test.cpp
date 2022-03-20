#include "hice/basic/slice.h"
#include "hice/basic/factories.h"
#include "hice/core/tensor.h"
#include "hice/core/tensor_printer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {
namespace {

using ::testing::Each;

#define INIT_VAL 0
#define NEW_VAL 1

template <typename TScalarType>
struct SliceTestParams {
  std::vector<int64_t> dims;
  int64_t axis;
  int64_t start;
  int64_t end;
};

template <typename TScalarType>
class SliceTest
    : public ::testing::TestWithParam<SliceTestParams<TScalarType>> {};

using SliceTestParamsFloat = SliceTestParams<float>;
using SliceTestFloat = SliceTest<float>;


TEST_P(SliceTestFloat, ShrinkedDimensions) {
  SliceTestParamsFloat params =
      ::testing::TestWithParam<SliceTestParamsFloat>::GetParam();
  int64_t axis = params.axis;
  int64_t start = params.start;
  int64_t end = params.end;
  // init all elements to 0.
  Tensor tensor = hice::full(params.dims, INIT_VAL, dtype(kFloat).device(kCPU));
  // assign the part to be sliced to 1.
  auto inner_size = tensor.size_from_dim(axis + 1);
  auto outer_size = tensor.size_to_dim(axis);
  auto outer_stride = inner_size * tensor.dim(axis);
  auto axis_stride = inner_size;
  auto offset = inner_size * (start - end);
  auto data_ptr = tensor.mutable_data<float>();
  for (int i = 0; i < outer_size; ++i) {
    for (int k = start; k < end; ++k) {
      for (int j = 0; j < inner_size; ++j) {
        data_ptr[i * outer_stride + k * axis_stride + j] = NEW_VAL;
      }
    }
  } 

  // for (int i = 0; i < 5; ++i) {
  //   for (int k = 0; k < 6; ++k) {
  //     for (int j = 0; j < 4; ++j) {
  //       std::cout << data_ptr[i * 24 + k * 4 + j] << ", ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }

  Tensor result = hice::slice(tensor, axis, start, end);
  EXPECT_EQ(result.size(), inner_size * outer_size * (end - start));
  hice::ArrayRef<float> result_data(result.mutable_data<float>(), result.size());
  EXPECT_THAT(result_data, Each(NEW_VAL));
}


// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    SliceTestFloatSuite, SliceTestFloat,
    ::testing::Values(
      // vector
      SliceTestParamsFloat{{5},
                            0,
                            1,
                            2},
      // matrix
      SliceTestParamsFloat{{5, 7},
                            1,
                            2,
                            3},
      // cube
      SliceTestParamsFloat{{5, 7, 10},
                            1,
                            2,
                            7}
    )
);

}  // namespace
}  // namespace hice