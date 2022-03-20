#include "hice/basic/copy.h"
#include "hice/core/expression_util.h"
#include "hice/math/binary_expr.h"
#include "hice/math/cpu/eval_expr_dense.h"
#include "hice/math/reduce.h"
#include "hice/nn/smooth_l1_loss.h"
#include "hice/util/math_utils.h"

namespace hice {

namespace {

template <typename scalar_t>
void smooth_loss_fwd_kernel(const Tensor &input, const Tensor &target,
                            Tensor &loss) {
  const scalar_t * input_data = input.data<scalar_t>();
  const scalar_t * target_data = target.data<scalar_t>();
  scalar_t * loss_data = loss.mutable_data<scalar_t>();
  int64_t size = loss.size();
  parallel_for(0, size, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      double err_abs = std::abs((double)input_data[i] - target_data[i]);
      loss_data[i] = err_abs < 1 ? 0.5 * err_abs * err_abs : err_abs - 0.5;
    }
  });
}

void smooth_l1_loss_fwd_impl(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight, Reduction reduction,
                             Tensor &loss) {
  Tensor tmp(input.dims(), device(input.device()).dtype(input.data_type()));
  HICE_DISPATCH_ALL_TYPES(input.scalar_type(), "smooth_l1_loss_fwd", [&]() {
    smooth_loss_fwd_kernel<scalar_t>(input, target, tmp);
  });
  if (weight) {
    mul(tmp, weight.value(), tmp);
  }
  switch (reduction) {
    case Reduction::none: {
      hice::copy(tmp, loss);
      break;
    }
    case Reduction::mean: {
      auto all_range = get_all_dims_range(tmp.dims());
      reduce_sum(tmp, all_range, false, loss);
      auto size = input.size();
      div(loss, size, loss);
      break;
    }
    case Reduction::sum: {
      auto all_range = get_all_dims_range(tmp.dims());
      reduce_sum(tmp, all_range, false, loss);
      break;
    }
    default: {
      HICE_LOG(ERROR) << "Not supported reduction type";
      break;
    }
  }
}

template <typename scalar_t>
void smooth_loss_bwd_kernel(const Tensor &input, const Tensor &target,
                            Tensor &grad_input) {
  const scalar_t * input_data = input.data<scalar_t>();
  const scalar_t * target_data = target.data<scalar_t>();
  scalar_t * grad_input_data = grad_input.mutable_data<scalar_t>();
  int64_t size = input.size();
  parallel_for(0, size, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      auto err = input_data[i] - target_data[i];
      if (-1 < err && err < 1) {
        grad_input_data[i] = err;
      } else if (err <= -1) {
        grad_input_data[i] = -1;
      } else {  // err >= 1
        grad_input_data[i] = 1;
      }
    }
  });
}

void smooth_l1_loss_bwd_impl(const Tensor &input, const Tensor &target,
                             hice::optional<Tensor> weight, Reduction reduction,
                             const Tensor &grad_loss, Tensor &grad_input) {
  Tensor tmp(input.dims(), device(input.device()).dtype(input.data_type()));
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "smooth_l1_loss_bwd", [&]() {
    smooth_loss_bwd_kernel<scalar_t>(input, target, grad_input);
  });
  mul(grad_input, grad_loss, grad_input);
  if (weight) {
    mul(grad_input, weight.value(), grad_input);
  }
  if (reduction == Reduction::mean) {
    auto size = input.size();
    div(grad_input, size, grad_input);
  }
}

}  // namespace

// Forward
HICE_REGISTER_KERNEL(smooth_l1_loss_fwd_dispatcher, &smooth_l1_loss_fwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // target
                     {kCPU, kDense}   // loss
);

// Backward
HICE_REGISTER_KERNEL(smooth_l1_loss_bwd_dispatcher, &smooth_l1_loss_bwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // target
                     {kCPU, kDense},  // grad_loss
                     {kCPU, kDense}   // grad_input
);

}  // namespace hice
