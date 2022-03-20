#include "hice/core/util.h"

#include "gtest/gtest.h"

namespace hice {
namespace {

TEST(UtilTest, CommonFactors) {
  struct {
    std::vector<int64_t> a, b;
    std::vector<std::pair<int64_t, int64_t>> expected;
  } test_cases[] = {
      {/*.a =*/{0}, /*.b =*/{0}, /*.expected =*/{{0, 0}, {1, 1}}},
      {/*.a =*/{}, /*.b =*/{}, /*.expected =*/{{0, 0}}},
      {/*.a =*/{2, 5, 1, 3},
       /*.b =*/{1, 10, 3, 1},
       /*.expected =*/{{0, 0}, {0, 1}, {2, 2}, {3, 2}, {4, 3}, {4, 4}}},
  };
  for (const auto& test_case : test_cases) {
    EXPECT_TRUE(hice::c_equal(test_case.expected,
                              common_factors(test_case.a, test_case.b)));
  }
}

}  // namespace
}  // namespace hice
