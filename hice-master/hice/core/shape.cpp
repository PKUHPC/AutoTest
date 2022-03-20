#include <iostream> 
#include "hice/core/shape.h"
#include "hice/core/shape_util.h"

namespace hice {

#if 0
Shape::Shape(ConstIntArrayRef dimensions) {
  dimensions_.reserve(dimensions.size());
  for (const int64_t dimension : dimensions) {
    add_dimensions(dimension);
  }
}
#endif

Shape::Shape(ConstIntArrayRef dimensions, const Layout& layout)
    : dimensions_(dimensions.begin(), dimensions.end()), layout_(layout) {
  HICE_CHECK_EQ(dimensions_.size(), layout_.minor_to_major_size());
}

std::string Shape::to_string(bool print_layout) const {
  if (print_layout) {
    return ShapeUtil::human_string_with_layout(*this);
  } else {
    return ShapeUtil::human_string(*this);
  }
}

void Shape::delete_dimension(int64_t dimension_to_delete) {
 HICE_CHECK_GE(dimension_to_delete, 0);
 HICE_CHECK_LT(dimension_to_delete, dimensions_.size());
  dimensions_.erase(dimensions_.begin() + dimension_to_delete);
  if (LayoutUtil::has_layout(*this)) {
    layout_.set_type(kDense);
    for (int64_t i = 0; i < layout_.minor_to_major().size();) {
      if (layout_.minor_to_major(i) == dimension_to_delete) {
        layout_.mutable_minor_to_major().erase(
            layout_.mutable_minor_to_major().begin() + i);
        continue;
      }
      if (layout_.minor_to_major(i) > dimension_to_delete) {
        layout_.mutable_minor_to_major()[i] -= 1;
      }
      ++i;
    }
  }
}

bool Shape::operator==(const Shape& other) const {
  if (this->layout().type() != other.layout().type()) {
    // HICE_LOG(INFO)
    //     << "Compare shapes: this layout type != other layout type";
    return false;
  }
  if (!ShapeUtil::same_dimensions(*this, other)) {
    // HICE_LOG(INFO) << "Compare shapes: this dimensions != other dimensions";
    return false;
  }
  if (!(LayoutUtil::minor_to_major(*this) ==
        LayoutUtil::minor_to_major(other))) {
    // HICE_LOG(INFO)
    //     << "Compare shapes: this layout order != other layout order";
    return false;
  }
  return true;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << shape.to_string(/*print_layout=*/true);
  return out;
}

std::ostream& operator<<(std::ostream& out, hice::ConstIntArrayRef arr)
{
  out << "[ ";
  for (int i = 0; i< arr.size(); ++i) {
    out << arr[i] << ", ";
  }
  out << "]\n";
  return out;
}

std::ostream& operator<<(std::ostream& out, std::vector<int64_t> arr)
{
  out << "[ ";
  for (int i = 0; i< arr.size(); ++i) {
    out << arr[i] << ", ";
  }
  out << "]\n";
  return out;
}

}  // namespace hice
