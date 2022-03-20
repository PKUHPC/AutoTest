#include "hice/nn/cross_entropy.h"
#include "hice/nn/softmax.h"
#include "hice/nn/softmax_cross_entropy.h"

namespace hice {

namespace {

void softmax_cross_entropy_fwd_impl(const Tensor &logit, const Tensor &target,
                                    hice::optional<Tensor> weight,
                                    const int64_t axis, Tensor &prob,
                                    Tensor &loss) {
  // std::cout << "cpu softmax cross entropy fwd" << std::endl;
  hice::softmax_fwd(logit, axis, prob);
  hice::cross_entropy_fwd(prob, target, weight, axis, loss);
}

void softmax_cross_entropy_bwd_impl(const Tensor &prob, const Tensor &target,
                                    hice::optional<Tensor> weight,
                                    const Tensor &grad_loss, const int64_t axis,
                                    Tensor &grad_logit) {
  // std::cout << "cpu softmax cross entropy bwd" << std::endl;
  int64_t true_axis = prob.get_true_axis(axis);
  int64_t outer_size = prob.size_to_dim(true_axis);
  int64_t axis_size = prob.dim(true_axis);
  int64_t inner_size = prob.size_from_dim(true_axis + 1);
  HICE_CHECK_EQ(compare_dims(prob.dims(), target.dims()), 0);
  HICE_CHECK_EQ(compare_dims(grad_logit.dims(), prob.dims()), 0);
  HICE_CHECK_EQ(grad_loss.size(), outer_size * inner_size);
  grad_logit.fill(0);
  ScalarType sc_type = prob.scalar_type();
  HICE_DISPATCH_ALL_TYPES(
      sc_type, "cpu_sparse_softmax_cross_entropy_bwd", [&]() {
        const scalar_t *prob_data = prob.data<scalar_t>();
        const scalar_t *target_data = target.data<scalar_t>();
        const scalar_t *grad_loss_data = grad_loss.data<scalar_t>();
        auto grad_logit_data = grad_logit.mutable_data<scalar_t>();
        const scalar_t *weight_data = nullptr;
        if (weight) {
          HICE_CHECK_EQ(compare_dims(grad_loss.dims(), weight.value().dims()),
                        0);
          weight_data = weight.value().data<scalar_t>();
        }
        for (int64_t index = 0; index < outer_size * inner_size; ++index) {
          int64_t inner_index = index % inner_size;
          int64_t outer_index = index / inner_size;
          float weight = (weight_data == nullptr ? 1.0 : weight_data[index]);
          for (int64_t c = 0; c < axis_size; ++c) {
            int64_t prob_index = outer_index * axis_size * inner_size +
                                 c * inner_size + inner_index;
            grad_logit_data[prob_index] =
                (prob_data[prob_index] - 1.0 * target_data[prob_index]) *
                grad_loss_data[index] * weight;
          }
        }
      });
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(softmax_cross_entropy_fwd_dispatcher,
                     &softmax_cross_entropy_fwd_impl, {kCPU, kDense},  // logit
                     {kCPU, kDense},                                   // target
                     {kCPU, kDense},                                   // prob
                     {kCPU, kDense}                                    // loss
);

HICE_REGISTER_KERNEL(softmax_cross_entropy_bwd_dispatcher,
                     &softmax_cross_entropy_bwd_impl, {kCPU, kDense},  // prob
                     {kCPU, kDense},                                   // target
                     {kCPU, kDense},  // grad_loss
                     {kCPU, kDense}   // grad_logit
);

}  // namespace hice