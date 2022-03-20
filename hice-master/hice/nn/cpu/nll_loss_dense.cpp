#include "hice/basic/memset.h"
#include "hice/core/expression_util.h"
#include "hice/math/binary_expr.h"
#include "hice/math/cpu/eval_expr_dense.h"
#include "hice/nn/activation.h"
#include "hice/nn/nll_loss.h"

namespace hice {

namespace {

template <typename scalar_t>
void nll_loss_fwd_kernel(const Tensor &input, const Tensor &target,
                         hice::optional<Tensor> weight, Tensor &loss) {
  auto input_data = input.data<scalar_t>();
  auto target_data = target.data<int64_t>();
  auto loss_data = loss.mutable_data<scalar_t>();
  const scalar_t *weight_data = nullptr;
  if (weight) {
    weight_data = weight.value().data<scalar_t>();
  }
  int64_t size_loss = loss.size();
  int64_t n_class = input.dim(-1);
  parallel_for(0, size_loss, hice::GRAIN_SIZE,
               [&](int64_t begin_loss, int64_t end_loss) {
                 int64_t idx_loss = begin_loss;
                 while (idx_loss < end_loss) {
                   int64_t trg = target_data[idx_loss];
                   auto offset_ipt = idx_loss * n_class + trg;
                   scalar_t ipt = input_data[offset_ipt];
                   scalar_t wgt = weight_data == nullptr ? 1 : weight_data[trg];
                   loss_data[idx_loss] = -1 * wgt * ipt;
                   ++idx_loss;
                 }
               });
}

void nll_loss_fwd_impl(const Tensor &input, const Tensor &target,
                       hice::optional<Tensor> weight, Tensor &loss) {
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "nll_loss_fwd_impl", [&]() {
    nll_loss_fwd_kernel<scalar_t>(input, target, weight, loss);
  });
}

template <typename scalar_t>
void nll_loss_bwd_kernel(const Tensor &input, const Tensor &target,
                         hice::optional<Tensor> weight, const Tensor &grad_loss,
                         Tensor &grad_input) {
  auto target_data = target.data<int64_t>();
  auto grad_loss_data = grad_loss.data<scalar_t>();
  auto grad_input_data = grad_input.mutable_data<scalar_t>();
  hice::memset(grad_input_data, 0, grad_input.size() * grad_input.item_size(),
         grad_input.device());
  const scalar_t *weight_data = nullptr;
  if (weight) {
    weight_data = weight.value().data<scalar_t>();
  }
  int64_t size_loss = grad_loss.size();
  int64_t n_class = input.dim(-1);
  parallel_for(0, size_loss, hice::GRAIN_SIZE,
               [&](int64_t begin_loss, int64_t end_loss) {
                 int64_t idx_loss = begin_loss;
                 while (idx_loss < end_loss) {
                   int64_t trg = target_data[idx_loss];
                   int64_t offset_ipt = idx_loss * n_class + trg;
                   scalar_t wgt = weight_data == nullptr ? 1 : weight_data[trg];
                   scalar_t grad_loss = grad_loss_data[idx_loss];
                   grad_input_data[offset_ipt] = -1 * wgt * grad_loss;
                   ++idx_loss;
                 }
               });
}

void nll_loss_bwd_impl(const Tensor &input, const Tensor &target,
                       hice::optional<Tensor> weight, const Tensor &grad_loss,
                       Tensor &grad_input) {
  ScalarType sc_type = input.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "nll_loss_bwd_impl", [&]() {
    nll_loss_bwd_kernel<scalar_t>(input, target, weight, grad_loss, grad_input);
  });
}

}  // namespace

// Forward
HICE_REGISTER_KERNEL(nll_loss_fwd_dispatcher, &nll_loss_fwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // target
                     {kCPU, kDense}   // loss
);

// Backward
HICE_REGISTER_KERNEL(nll_loss_bwd_dispatcher, &nll_loss_bwd_impl,
                     {kCPU, kDense},  // input
                     {kCPU, kDense},  // target
                     {kCPU, kDense},  // grad_loss
                     {kCPU, kDense}   // grad_input
);

}  // namespace hice
