#include "hice/basic/factories.h"
#include "hice/device/cpu/context_cpu.h"
#include "hice/nn/dropout.h"

namespace hice {

namespace {

void dropout_fwd_impl(Tensor &input, double rate, Tensor &mask,
                      Tensor &output) {
  HICE_CHECK_GE(1.0, rate) << "rate must be <= 1.0";
  HICE_CHECK_GE(rate, 0.0) << "rate must be >= 0.0";
  HICE_CHECK_EQ(input.ndim(), mask.ndim())
      << "Dimensions of mask and input  must be equal";
  if (rate == 1) {
    output.fill(0);
    mask.fill(0);
    return;
  }
  ScalarType sc_type = input.scalar_type();
  auto data_ptr_mask = mask.mutable_data<bool>();
  CPUContext cpu_ctx;
  auto rand_generator = cpu_ctx.rand_generator();
  std::bernoulli_distribution b_distribution(1 - rate);
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "DROPOUT", [&]() {
    auto data_ptr_in = input.data<scalar_t>();
    auto data_ptr_out = output.mutable_data<scalar_t>();
    for (int i = 0; i < input.size(); i++) {
      data_ptr_mask[i] = b_distribution(rand_generator);
      data_ptr_out[i] = data_ptr_in[i] * data_ptr_mask[i] / (1 - rate);
    }
  });
}
void dropout_bwd_impl(Tensor &input, double rate, Tensor &mask,
                      Tensor &output) {
  HICE_CHECK_GE(1.0, rate) << "rate must be <= 1.0";
  HICE_CHECK_GE(rate, 0.0) << "rate must be >= 0.0";
  HICE_CHECK_EQ(input.ndim(), mask.ndim())
      << "Dimensions of mask and input  must be equal";
  if (rate == 1) {
    output.fill(0);
    return;
  }
  ScalarType sc_type = input.scalar_type();
  const bool *data_ptr_mask = mask.data<bool>();
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "DROPOUT", [&]() {
    const scalar_t *data_ptr_in = input.data<scalar_t>();
    auto data_ptr_out = output.mutable_data<scalar_t>();
    for (int i = 0; i < input.size(); i++) {
      data_ptr_out[i] = data_ptr_in[i] * data_ptr_mask[i] / (1 - rate);
    }
  });
}
}  // namespace

HICE_REGISTER_KERNEL(dropout_fwd_dispatcher, &dropout_fwd_impl, {kCPU, kDense},
                     {kCPU, kDense}, {kCPU, kDense});
HICE_REGISTER_KERNEL(dropout_bwd_dispatcher, &dropout_bwd_impl, {kCPU, kDense},
                     {kCPU, kDense}, {kCPU, kDense});

}  // namespace hice
