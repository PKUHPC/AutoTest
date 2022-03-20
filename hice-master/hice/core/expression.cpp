#include "expression.h"
#include "expression_util.h"

namespace hice {

void Expression::resize_output() {
  std::vector<int64_t> dims_out_new;
  switch (type_) {
    case ExprType::kUnary:
      ExpressionUtil::may_resize_result(output(0), input(0).dims(), output_resizable_);
      break;
    case ExprType::kBinary:
      dims_out_new = hice::broadcast(input(0).dims(), input(1).dims());
      ExpressionUtil::may_resize_result(output(0), dims_out_new, output_resizable_);
      break;
    default:
      HICE_CHECK(false) << "This function is only for unary and binary.";
      break;
  }
}

void Expression::resize_output(std::vector<bool> mask, bool keep_dim) {
  std::vector<int64_t> dims(input(0).shape().dimensions());
  auto ndim = dims.size();
  for (int i = ndim - 1; i >= 0; i--) {
    if (mask[i]) {
      if (keep_dim) {
        dims[i] = 1;
      } else {
        dims.erase(dims.begin() + i);
      }
    }
  }
  output(0).resize(dims);
}

void Expression::review_reduction_result(std::vector<bool> mask,
                                         bool keep_dim) {
  int ndim = input(0).ndim();
  if (!keep_dim) {
    std::vector<int64_t> dims(output(0).shape().dimensions());
    for (int i = 0; i < ndim; i++) {
      if (mask[i]) {
        dims.insert(dims.begin() + i, 1);
      }
    }
    Tensor reviewed_result(
        device(output(0).device()).dtype(output(0).data_type()));
    reviewed_result.resize(dims);
    this->reviewed_outputs_.emplace_back(reviewed_result);
  }
}

void Expression::prepare_strides() {
  const Tensor& in0 = input(0);
  Tensor out;
  if (reviewed_outputs_.size() != 0) {
    out = reviewed_output(0);
  } else {
    out = output(0);
  }
  size_t length_stride = std::max(in0.ndim(), out.ndim());
  for (int i = 0; i < num_inputs(); ++i) {
    const Tensor& in = input(i);
    strides_inputs_.emplace_back(
        ExpressionUtil::strides_for_computing(in.strides(), in.dims(), length_stride));
  }
  strides_outputs_.emplace_back(
      ExpressionUtil::strides_for_computing(out.strides(), out.dims(), length_stride));
}

std::vector<bool> Expression::make_reduction_mask(
    ConstIntArrayRef reduced_dims) {
  int ndim = input(0).ndim();
  std::vector<bool> mask(ndim, false);
  for (int i = 0; i < reduced_dims.size(); i++) {
    mask[reduced_dims[i]] = true;
  }
  return mask;
}

std::vector<int64_t> Expression::reorder_dims(std::vector<int64_t> strides) {
  int ndim = strides.size();
  std::vector<int64_t> perm_(ndim, 0);
  // ndim-1, ndim-2, ..., 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);
  auto should_swap = [&](int64_t dim0, int64_t dim1) {
    return strides[dim0] < strides[dim1];
  };
  for (int i = 1; i < ndim; i++) {
    for (int j = i; j > 0; j--) {
      bool comparison = should_swap(perm_[j], perm_[j - 1]);
      if (comparison) {
        std::swap(perm_[j], perm_[j - 1]);
      } else {
        break;
      }
    }
  }
  return perm_;
}

std::vector<int64_t> Expression::permute_dims(const std::vector<int64_t> in,
                                              std::vector<int64_t> perm) {
  int ndim = in.size();
  std::vector<int64_t> out(ndim, 0);
  for (int i = 0; i < ndim; i++) {
    out[i] = in[perm[i]];
  }
  return out;
}

}  // namespace hice