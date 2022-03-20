#include "hice/util/string_ops.h"
#include "hice/util/loguru.h"
#include "hice/core/layout.h"

namespace hice {

namespace {

std::string to_string(LayoutType type) {
  switch (type) {
    case kDense:
      return "dense";
    case kCOO:
      return "coo";
    case kCSR:
      return "csr";
    default:
      HICE_CHECK_EQ(type, LayoutType::Invalid);
      return "invalid";
  }
}

std::string to_string(LayoutOrder f, bool lower_case) {
  switch (f) {
    case LayoutOrder::NCHW:
      return lower_case ? "nchw" : "NCHW";
    case LayoutOrder::NHWC:
      return lower_case ? "nhwc" : "NHWC";
    default:
      HICE_LOG(ERROR) << "Unknown format order: " << static_cast<int16_t>(f);
      return "";
  }
}

}  // anonymous namespace

std::ostream& operator<<(std::ostream& stream, LayoutType type) {
  stream << to_string(type);
  return stream;
}

std::ostream& operator<<(std::ostream& stream, LayoutOrder type) {
  stream << to_string(type, /* lower case */ true);
  return stream;
}

std::string Layout::to_string() const {
  if (type() == kDense) {
    return StrCat("{", StrJoin(minor_to_major(), ","), "}");
  } else if (type() == kCOO) {
    return StrCat("coo{", 0,"}");
  } else if (type() == kCSR) {
    return StrCat("csr{", 0,"}");
  } else {
    HICE_CHECK_EQ(type(), LayoutType::Invalid);
    return "invalid{}";
  }
}

bool Layout::operator==(const Layout& other) const {
  return (other.type() == type() &&
          other.minor_to_major() == minor_to_major());
}

std::ostream& operator<<(std::ostream& out, const Layout& layout) {
  out << layout.to_string();
  return out;
}

} // namespace hice 