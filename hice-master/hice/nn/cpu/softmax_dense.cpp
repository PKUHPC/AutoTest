#ifdef HICE_USE_CPU_NATIVE

#include "hice/nn/softmax.h"
#include "hice/math/reduce.h"
#include "hice/math/binary_expr.h"
#include "hice/math/unary_expr.h"

namespace hice {

namespace {

void softmax_fwd_impl(const Tensor &input, int64_t axis, Tensor &output) {
  // max_logits
  Tensor tmp_max = reduce_max(input, {axis}, true);
  // logits - max_logits 
  Tensor tmp_sub = sub(input, tmp_max);
  // exp(logits - max_logits)
  Tensor tmp_exp = exp(tmp_sub);
  // sum_{class}(exp(logits - max_logits))
  Tensor tmp_sum = reduce_sum(tmp_exp, {axis}, true);
  // exp(logits -max_logits) / sum_{class}(exp(logits - max_logits))
  div(tmp_exp, tmp_sum, output);
}


void softmax_bwd_impl(const Tensor &output, const Tensor &grad_output,
                      int64_t axis, Tensor &grad_input) {
  // dy * y
  Tensor tmp_mul = mul(grad_output, output);
  // sum_{class}(dy * y)
  Tensor tmp_sum = reduce_sum(tmp_mul, {axis}, true); 
  // dy - sum_{class}(dy * y) 
  Tensor tmp_sub = sub(grad_output, tmp_sum); 
  // y * (dy - sum_{class}(dy * y))
  mul(output, tmp_sub, grad_input); 
}

} // anonymous namespace

HICE_REGISTER_KERNEL(
    softmax_fwd_dispatcher, 
    &softmax_fwd_impl, 
    {kCPU, kDense}, // input 
    {kCPU, kDense}  // output 
);

HICE_REGISTER_KERNEL(
    softmax_bwd_dispatcher, 
    &softmax_bwd_impl, 
    {kCPU, kDense}, // output 
    {kCPU, kDense},  // grad_output 
    {kCPU, kDense}  // grad_input
);

} // namespace hice

#endif