#include <hice/tvm/tvm.h>
#include <hice/tvm/operators.h>
#include <hice/util/benchmark.h>

#include <hice/intelligent/dense_plan.h>
// #include <hice/nn/dense.h>

namespace hice {

void DensePlan::evaluate() {
  if (!is_tvm_available()) HICE_LOG(ERROR) << "tvm is not enabled";
  if (is_evaluated_) return;
  
  // no hice kernel, use tvm

  bool success = dense_fwd_tvm(input(0), input(1), input(2), output(0));

  impl_type_ = success ? kTVM : kOfficial;
  is_evaluated_ = true;
}

void DensePlan::execute() {
  if (impl_type_ == kTVM) {
    HICE_DLOG(INFO) << "execute tvm kernel.";
    dense_fwd_tvm(input(0), input(1), input(2), output(0));
  } else {
    HICE_DLOG(ERROR) << "Dense hice kernel is not aviliable.";
  }
}


void DenseGradPlan::evaluate() {
  if (!is_tvm_available()) HICE_LOG(ERROR) << "tvm is not enabled";
  if (is_evaluated_) return;
  
  // no hice kernel, use tvm
  bool success = dense_bwd_tvm(input(0), input(1), input(2), output(0), output(1), output(2));
  
  impl_type_ = success ? kTVM : kOfficial;
  is_evaluated_ = true;
}

void DenseGradPlan::execute() {
  if (impl_type_ == kTVM) {
    HICE_DLOG(INFO) << "execute tvm kernel.";
    dense_bwd_tvm(input(0), input(1), input(2), output(0), output(1), output(2));
  } else {
    HICE_DLOG(ERROR) << "Dense hice kernel is not aviliable.";
  }
}

} // namespace hice