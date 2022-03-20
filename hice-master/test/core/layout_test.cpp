#include "hice/core/layout.h"

#include "gtest/gtest.h"

namespace hice {
namespace {

class LayoutTest : public ::testing::Test {};

TEST_F(LayoutTest, LayoutToString) {
  EXPECT_EQ(Layout().to_string(), "invalid{}");
  EXPECT_EQ(Layout({4, 5, 6}).to_string(), "{4,5,6}");
}

TEST_F(LayoutTest, StreamOut) {
  {
    std::ostringstream oss;
    oss << Layout({0, 1, 2});
    EXPECT_EQ(oss.str(), "{0,1,2}");
  }
}

TEST_F(LayoutTest, Equality) {
  EXPECT_EQ(Layout(), Layout());
  const std::vector<int64_t> empty_dims;
  EXPECT_EQ(Layout(empty_dims), Layout(empty_dims));
  EXPECT_NE(Layout(), Layout(empty_dims));
  EXPECT_EQ(Layout({0, 1, 2, 3}), Layout({0, 1, 2, 3}));
  EXPECT_NE(Layout({0, 1, 2, 3}), Layout({0, 1, 2}));
}

}  // namespace
}  // namespace hice
