
#include "hice/api_c/ops_math.h"
#include "hice/api_c/tensor_impl.h"
#include "hice/api_c/error_handle.h"

#include "hice/math/matmul.h"
#include "hice/math/reduce.h"
#include "hice/math/binary_expr.h"
#include "hice/math/compare.h"
#include "hice/math/unary_expr.h"
#include "hice/util/loguru.h"

HI_Status HI_Matmul(const HI_Tensor tensor1, const HI_Tensor tensor2,
               HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::matmul(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}

HI_Status HI_Reduce(const HI_Tensor input, HI_ReduceMode mode, const int64_t *axis,
               int64_t axis_len, bool keepdim, HI_Tensor *output) {
  HI_API_BEGIN();
  hice::ConstIntArrayRef new_dim(&axis[0], axis_len);
  switch (mode) {
    case SUM:
      *output = new HI_Tensor_Impl{hice::reduce_sum(input->tensor_, new_dim, keepdim)}; 
      break;
    case PROD:
      *output = new HI_Tensor_Impl{hice::reduce_prod(input->tensor_, new_dim, keepdim)}; 
      break;
    case MEAN:
      *output = new HI_Tensor_Impl{hice::reduce_mean(input->tensor_, new_dim, keepdim)}; 
      break; 
    case MAX:
      *output = new HI_Tensor_Impl{hice::reduce_max(input->tensor_, new_dim, keepdim)}; 
      break;
    case MIN:
      *output = new HI_Tensor_Impl{hice::reduce_min(input->tensor_, new_dim, keepdim)}; 
      break; 
    default:
      break;
  }
  HI_API_END();
}

HI_Status HI_Add(const HI_Tensor tensor1, const HI_Tensor tensor2,
                 HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::add(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}
HI_Status HI_Sub(const HI_Tensor tensor1, const HI_Tensor tensor2,
                 HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::sub(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}
HI_Status HI_Mul(const HI_Tensor tensor1, const HI_Tensor tensor2,
                 HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::mul(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}
HI_Status HI_Div(const HI_Tensor tensor1, const HI_Tensor tensor2,
                 HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::div(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}

HI_Status HI_Exp(const HI_Tensor input, HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::exp(input->tensor_)};
  HI_API_END();
}
HI_Status HI_Log(const HI_Tensor input, HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::log(input->tensor_)};
  HI_API_END();
}
HI_Status HI_Neg(const HI_Tensor input, HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::neg(input->tensor_)};
  HI_API_END();
}

HI_Status HI_Equal(const HI_Tensor tensor1, const HI_Tensor tensor2,
                   HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::equal(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}
HI_Status HI_Less(const HI_Tensor tensor1, const HI_Tensor tensor2,
                  HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{hice::less(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}
HI_Status HI_LessEqual(const HI_Tensor tensor1, const HI_Tensor tensor2,
                        HI_Tensor *output) {
  HI_API_BEGIN();
  *output =
      new HI_Tensor_Impl{hice::less_equal(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}
HI_Status HI_Greater(const HI_Tensor tensor1, const HI_Tensor tensor2,
                     HI_Tensor *output) {
  HI_API_BEGIN();
  *output =
      new HI_Tensor_Impl{hice::greater(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}
HI_Status HI_GreaterEqual(const HI_Tensor &tensor1, const HI_Tensor &tensor2,
                           HI_Tensor *output) {
  HI_API_BEGIN();
  *output = new HI_Tensor_Impl{
      hice::greater_equal(tensor1->tensor_, tensor2->tensor_)};
  HI_API_END();
}

#if 1
    HI_Status HI_Matmul_Inplace(const HI_Tensor tensor1,
                                const HI_Tensor tensor2, HI_Tensor output) {
  HI_API_BEGIN();
  hice::matmul(tensor1->tensor_, tensor2->tensor_, output->tensor_);
  HI_API_END();
}
#endif