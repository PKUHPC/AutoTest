#include "hice/basic/cpu/index_helper.h"
#include "hice/nn/sparse_cross_entropy.h"
#include "hice/util/eigen.h"

namespace hice {

namespace {

template <typename T>
constexpr T log_threshold() {
  return static_cast<T>(1e-20);
}

void sparse_cross_entropy_fwd_impl(const Tensor &prob, const Tensor &target,
                                   hice::optional<Tensor> weight,
                                   const int64_t axis, Tensor &loss) {
  // std::cout << "cpu sparse cross entropy fwd" << std::endl;
  int64_t true_axis = prob.get_true_axis(axis);
  int64_t outer_size = prob.size_to_dim(true_axis);
  int64_t axis_size = prob.dim(true_axis);
  int64_t inner_size = prob.size_from_dim(true_axis + 1);
  HICE_CHECK_EQ(target.scalar_type(), kInt64);
  HICE_CHECK_EQ(target.ndim(), prob.ndim() - 1);
  HICE_CHECK_EQ(target.size(), outer_size * inner_size);
  HICE_CHECK_EQ(compare_dims(target.dims(), loss.dims()), 0);
  ScalarType sc_type = prob.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "cpu_sparse_cross_entropy_fwd", [&]() {
    const scalar_t *prob_data = prob.data<scalar_t>();
    const int64_t *target_data = target.data<int64_t>();
    scalar_t *loss_data = loss.mutable_data<scalar_t>();
    const scalar_t *weight_data = nullptr;
    if (weight) {
      HICE_CHECK_EQ(compare_dims(loss.dims(), weight.value().dims()), 0);
      weight_data = weight.value().data<scalar_t>();
    }
    for (int64_t index = 0; index < outer_size * inner_size; ++index) {
      int64_t inner_index = index % inner_size;
      int64_t outer_index = index / inner_size;
      int64_t target = static_cast<int64_t>(target_data[index]);
      float weight = (weight_data == nullptr ? 1.0 : weight_data[index]);
      int64_t prob_index = outer_index * axis_size * inner_size +
                           target * inner_size + inner_index;
      loss_data[index] = -std::log(std::max(prob_data[prob_index],
                                            log_threshold<scalar_t>())) *
                         weight;
    }
  });
}

void sparse_cross_entropy_bwd_impl(const Tensor &prob, const Tensor &target,
                                   hice::optional<Tensor> weight,
                                   const Tensor &grad_loss, const int64_t axis,
                                   Tensor &grad_prob) {
  // std::cout << "cpu sparse cross entropy bwd" << std::endl;
  int64_t true_axis = prob.get_true_axis(axis);
  int64_t outer_size = prob.size_to_dim(true_axis);
  int64_t axis_size = prob.dim(true_axis);
  int64_t inner_size = prob.size_from_dim(true_axis + 1);
  HICE_CHECK_EQ(target.scalar_type(), kInt64);
  HICE_CHECK_EQ(target.ndim(), prob.ndim() - 1);
  HICE_CHECK_EQ(target.size(), outer_size * inner_size);
  HICE_CHECK_EQ(compare_dims(target.dims(), grad_loss.dims()), 0);
  grad_prob.fill(0);
  ScalarType sc_type = prob.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "cpu_sparse_cross_entropy_bwd", [&]() {
    const scalar_t *prob_data = prob.data<scalar_t>();
    const int64_t *target_data = target.data<int64_t>();
    const scalar_t *grad_loss_data = grad_loss.data<scalar_t>();
    scalar_t *grad_prob_data = grad_prob.mutable_data<scalar_t>();
    const scalar_t *weight_data = nullptr;
    if (weight) {
      HICE_CHECK_EQ(compare_dims(grad_loss.dims(), weight.value().dims()), 0);
      weight_data = weight.value().data<scalar_t>();
    }
    for (int64_t index = 0; index < outer_size * inner_size; ++index) {
      int64_t inner_index = index % inner_size;
      int64_t outer_index = index / inner_size;
      int64_t target = static_cast<int64_t>(target_data[index]);
      float weight = (weight_data == nullptr ? 1.0 : weight_data[index]);
      int64_t prob_index = outer_index * axis_size * inner_size +
                           target * inner_size + inner_index;
      grad_prob_data[prob_index] =
          -(grad_loss_data[index] /
            std::max(prob_data[prob_index], log_threshold<scalar_t>())) *
          weight;
    }
  });
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(sparse_cross_entropy_fwd_dispatcher,
                     &sparse_cross_entropy_fwd_impl, {kCPU, kDense},  // prob
                     {kCPU, kDense},                                  // target
                     {kCPU, kDense}                                   // loss
);

HICE_REGISTER_KERNEL(sparse_cross_entropy_bwd_dispatcher,
                     &sparse_cross_entropy_bwd_impl, {kCPU, kDense},  // prob
                     {kCPU, kDense},                                  // target
                     {kCPU, kDense},  // grad_loss
                     {kCPU, kDense}   // grad_prob
);

}  // namespace hice