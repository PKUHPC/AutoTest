#pragma once

#include <ostream>
#include <vector>

#include "hice/util/types.h"
#include "hice/core/macros.h"

namespace hice {

enum class LayoutType : int16_t {
  Dense = 0,
  COO = 1,
  CSR = 2,
  Invalid = 3,
  NumLayoutTypes,
};
constexpr LayoutType kDense = LayoutType::Dense;
constexpr LayoutType kCOO = LayoutType::COO;
constexpr LayoutType kCSR = LayoutType::CSR;
constexpr LayoutType kInvalid = LayoutType::Invalid;
constexpr int kNumLayoutTypes = static_cast<int>(LayoutType::NumLayoutTypes);
HICE_API std::ostream& operator<<(std::ostream& stream, LayoutType type);

enum class LayoutOrder : int16_t {
  NHWC = 0,
  NCHW = 1,
  NumLayoutOrders,
};
constexpr LayoutOrder kNHWC = LayoutOrder::NHWC;
constexpr LayoutOrder kNCHW = LayoutOrder::NCHW;
constexpr int kNumLayoutOrders = static_cast<int>(LayoutOrder::NumLayoutOrders);
HICE_API std::ostream& operator<<(std::ostream& stream, LayoutOrder type);

struct HICE_API Layout {
  Layout() = default;

  Layout(LayoutType type) : type_(type), minor_to_major_() {}

  Layout(ConstIntArrayRef minor_to_major)
      : type_(kDense),
        minor_to_major_(minor_to_major.begin(), minor_to_major.end()) {}

  Layout(LayoutType type, ConstIntArrayRef minor_to_major)
      : type_(type),
        minor_to_major_(minor_to_major.begin(), minor_to_major.end()) {}

  // Returns a human-readable string that represents this layout.
  std::string to_string() const;

  bool operator==(const Layout& other) const;
  bool operator!=(const Layout& other) const { return !(*this == other); }

  LayoutType type() const noexcept { return type_; }
  void set_type(LayoutType type) noexcept { type_ = type; }

  // Methods for accessing the minor-to-major array.
  int minor_to_major_size() const { return minor_to_major_.size(); }
  int64_t minor_to_major(int index) const { return minor_to_major_.at(index); }
  Layout& set_minor_to_major(int index, int64_t value) {
    minor_to_major_.at(index) = value;
    return *this;
  }
  Layout& add_minor_to_major(int64_t value) {
    minor_to_major_.push_back(value);
    return *this;
  }
  Layout& clear_minor_to_major() {
    minor_to_major_.clear();
    return *this;
  }
  const std::vector<int64_t>& minor_to_major() const { return minor_to_major_; }
  std::vector<int64_t>& mutable_minor_to_major() { return minor_to_major_; }

  void swap(Layout* other) noexcept { std::swap(*this, *other); }

  void clear() {
    type_ = LayoutType::Invalid;
    minor_to_major_.clear();
  }

 private:
  LayoutType type_ = LayoutType::Invalid;
  // Sequence of logical dimension numbers, from minor (fastest varying dim)
  // to to major (slowest varying dim)
  std::vector<int64_t> minor_to_major_;
};

HICE_API std::ostream& operator<<(std::ostream& os, const Layout& layout);

} // namespace hice