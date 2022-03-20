#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/ml/dataset.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/tools/compare.h"

#include <tuple>

namespace hice {
using ::testing::Each;

#if 0 
TEST(DatasetTest, DenseFloat) {
  std::tuple<Tensor, Tensor, Tensor, Tensor> res; 
  res = load_dataset("a1a");
  Tensor ref = std::get<0>(res);
  Tensor query = std::get<1>(res);
  Tensor ref_label = std::get<2>(res);
  Tensor query_label = std::get<3>(res);
  EXPECT_EQ(ref.dim(0), 1605);
  EXPECT_EQ(ref.dim(1), 123); 
  EXPECT_EQ(query.dim(0), 30956);
  EXPECT_EQ(query.dim(1), 123); 
}
#endif

}  // namespace hice