#include "expression_util.h"

namespace hice {

Expression ExpressionUtil::make_unary_expr(const Tensor& in, Tensor& out,
                                                  bool resizable) {
  // DO NOT change the order of following codes
  Expression expr;
  expr.add_input(in);
  expr.add_output(out);
  expr.set_type(ExprType::kUnary);
  expr.set_resizable(resizable);
  expr.resize_output();
  expr.prepare_strides();
  return expr;
}

Expression ExpressionUtil::make_binary_expr(const Tensor& in1,
                                                   const Tensor& in2,
                                                   Tensor& out,
                                                   bool resizable) {
  Expression expr;
  expr.add_input(in1);
  expr.add_input(in2);
  expr.add_output(out);
  expr.set_type(ExprType::kBinary);
  expr.set_resizable(resizable);
  expr.resize_output();
  expr.prepare_strides();
  return expr;
}

Expression ExpressionUtil::make_reduction_expr(
  const Tensor& in, Tensor& out, ConstIntArrayRef reduced_dims, bool keep_dim,
  bool resizable) {
  // DO NOT change the order of following codes
  HICE_CHECK(in.is_default_layout());
  HICE_CHECK(out.is_default_layout());
  Expression expr;
  expr.add_input(in);
  expr.add_output(out);
  expr.set_type(ExprType::kReduction);
  expr.set_resizable(resizable);
  auto mask = expr.make_reduction_mask(reduced_dims);
  expr.resize_output(mask, keep_dim);
  expr.review_reduction_result(mask, keep_dim);
  expr.prepare_strides();
  return expr;
}

void ExpressionUtil::may_resize_result(Tensor& output,
                                              ConstIntArrayRef dims_output,
                                              bool resizable) {
  if (resizable) {
    output.resize(dims_output);
  } else {
    HICE_CHECK(output.dims() == dims_output)
      << "Invalid dims for result tensor";
  }
}

std::vector<int64_t> ExpressionUtil::strides_for_computing(
  ConstIntArrayRef strides_old, ConstIntArrayRef dims,
  int64_t length_stride) {
  size_t ndim_self = dims.size();
  int64_t offset = length_stride - ndim_self;
  std::vector<int64_t> strides_new(length_stride, 0);
  for (size_t j = 0; j < ndim_self; ++j) {
    if (dims[j] != 1) {
      strides_new[j + offset] = strides_old[j];
    }
  }
  return strides_new;
}

std::vector<int64_t> ExpressionUtil::minor_to_major_for_computing(
  ConstIntArrayRef minor_to_major_old, int64_t length) {
  std::vector<int64_t> minor_to_major_new = std::vector<int64_t>(length, 0);
  int64_t size_old = minor_to_major_old.size();
  int64_t offset = length - size_old;
  // copy old order to the end of the array
  for (size_t j = 0; j < size_old; ++j) {
    minor_to_major_new[j + offset] = minor_to_major_old[j];
  }
  // set the rest order as default
  for (size_t j = 0; j < offset; ++j) {
    minor_to_major_new[j] = length - 1 - j;
  }
  return minor_to_major_new;
}

}  // namespace hice