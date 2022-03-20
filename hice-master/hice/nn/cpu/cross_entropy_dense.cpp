#include "hice/basic/cpu/index_helper.h"
#include "hice/nn/cross_entropy.h"
#include "hice/util/eigen.h"

namespace hice {

namespace {

template <typename T>
constexpr T log_threshold() {
  return static_cast<T>(1e-20);
}

void cross_entropy_fwd_impl(const Tensor &prob, const Tensor &target,
                            hice::optional<Tensor> weight, const int64_t axis,
                            Tensor &loss) {
  // std::cout << "cpu cross entropy fwd" << std::endl;
  int64_t true_axis = prob.get_true_axis(axis);
  int64_t outer_size = prob.size_to_dim(true_axis);
  int64_t axis_size = prob.dim(true_axis);
  int64_t inner_size = prob.size_from_dim(true_axis + 1);
  HICE_CHECK_EQ(compare_dims(prob.dims(), target.dims()), 0);
  HICE_CHECK_EQ(loss.size(), outer_size * inner_size);
  ScalarType sc_type = prob.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "cpu_cross_entropy_fwd", [&]() {
    const scalar_t *prob_data = prob.data<scalar_t>();
    const scalar_t *target_data = target.data<scalar_t>();
    auto loss_data = loss.mutable_data<scalar_t>();
    scalar_t *weight_data = nullptr;
    if (weight) {
      HICE_CHECK_EQ(compare_dims(loss.dims(), weight.value().dims()), 0);
      weight_data = weight.value().mutable_data<scalar_t>();
    }
    for (int64_t index = 0; index < outer_size * inner_size; ++index) {
      int64_t inner_index = index % inner_size;
      int64_t outer_index = index / inner_size;
      float weight = (weight_data == nullptr ? 1.0 : weight_data[index]);
      loss_data[index] = 0;
      for (int64_t c = 0; c < axis_size; ++c) {
        int64_t prob_index =
            outer_index * axis_size * inner_size + c * inner_size + inner_index;
        loss_data[index] += -target_data[prob_index] *
                            std::log(std::max(prob_data[prob_index],
                                              log_threshold<scalar_t>()));
      }
      loss_data[index] *= weight;
    }
  });
}

void cross_entropy_bwd_impl(const Tensor &prob, const Tensor &target,
                            hice::optional<Tensor> weight,
                            const Tensor &grad_loss, const int64_t axis,
                            Tensor &grad_prob) {
  // std::cout << "cpu cross entropy bwd" << std::endl;
  int64_t true_axis = prob.get_true_axis(axis);
  int64_t outer_size = prob.size_to_dim(true_axis);
  int64_t axis_size = prob.dim(true_axis);
  int64_t inner_size = prob.size_from_dim(true_axis + 1);
  HICE_CHECK_EQ(compare_dims(prob.dims(), target.dims()), 0);
  HICE_CHECK_EQ(compare_dims(prob.dims(), grad_prob.dims()), 0);
  HICE_CHECK_EQ(grad_loss.size(), outer_size * inner_size);
  grad_prob.fill(0);
  ScalarType sc_type = prob.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "cpu_cross_entropy_bwd", [&]() {
    auto prob_data = prob.data<scalar_t>();
    auto target_data = target.data<scalar_t>();
    auto grad_loss_data = grad_loss.data<scalar_t>();
    auto grad_prob_data = grad_prob.mutable_data<scalar_t>();
    scalar_t *weight_data = nullptr;
    if (weight) {
      HICE_CHECK_EQ(compare_dims(grad_loss.dims(), weight.value().dims()), 0);
      weight_data = weight.value().mutable_data<scalar_t>();
    }
    for (int64_t index = 0; index < outer_size * inner_size; ++index) {
      int64_t inner_index = index % inner_size;
      int64_t outer_index = index / inner_size;
      float weight = (weight_data == nullptr ? 1.0 : weight_data[index]);
      for (int64_t c = 0; c < axis_size; ++c) {
        int64_t prob_index =
            outer_index * axis_size * inner_size + c * inner_size + inner_index;
        grad_prob_data[prob_index] =
            -(grad_loss_data[index] * weight) *
            (target_data[prob_index] /
             std::max(prob_data[prob_index], log_threshold<scalar_t>()));
      }
    }
  });
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(cross_entropy_fwd_dispatcher, &cross_entropy_fwd_impl,
                     {kCPU, kDense},  // prob
                     {kCPU, kDense},  // target
                     {kCPU, kDense}   // loss
);

HICE_REGISTER_KERNEL(cross_entropy_bwd_dispatcher, &cross_entropy_bwd_impl,
                     {kCPU, kDense},  // prob
                     {kCPU, kDense},  // target
                     {kCPU, kDense},  // grad_loss
                     {kCPU, kDense}   // grad_prob
);

}  // namespace hice