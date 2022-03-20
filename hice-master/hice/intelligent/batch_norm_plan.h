#pragma once

#include <hice/intelligent/plan.h>
#include <hice/util/types.h>

namespace hice {
// saved_rvars = sqrt(var + eps)
class HICE_API BatchNormPlan : public Plan {
public:
  BatchNormPlan(const DLTensor& input, 
          const DLTensor& scale,
          const DLTensor& bias,
          const DLTensor& running_mean,
          const DLTensor& running_var,
          double momentum,
          double eps,
          DLTensor& output,
          DLTensor& saved_mean,
          DLTensor& saved_rvars): momentum_(momentum), eps_(eps) {
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_IN(scale);
    REGISTER_PLAN_IN(bias);
    REGISTER_PLAN_IN(running_mean);
    REGISTER_PLAN_IN(running_var);
    Tensor& in0 = this->input(0);
    Tensor tensor_momentum = empty({1}, device(in0.device()).dtype(in0.data_type()));
    Tensor tensor_eps = empty({1}, device(in0.device()).dtype(in0.data_type()));
    tensor_momentum.fill(momentum);
    tensor_eps.fill(eps);
    REGISTER_PLAN_IN_HICETENSOR(tensor_momentum);
    REGISTER_PLAN_IN_HICETENSOR(tensor_eps);
    REGISTER_PLAN_OUT(output);
    REGISTER_PLAN_OUT(saved_mean);
    REGISTER_PLAN_OUT(saved_rvars);
  }

  virtual void evaluate();
  virtual void execute();

private:
  double momentum_;
  double eps_;
};

class HICE_API BatchNormGradPlan : public Plan {
public:
  BatchNormGradPlan(const DLTensor& input, 
                    const DLTensor& scale,
                    const DLTensor& bias,
                    const DLTensor& saved_mean,
                    const DLTensor& saved_var,
                    const DLTensor& grad_output,
                    double eps,
                    DLTensor& grad_input,
                    DLTensor& grad_scale,
                    DLTensor& grad_bias): eps_(eps) {
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_IN(scale);
    REGISTER_PLAN_IN(bias);
    REGISTER_PLAN_IN(saved_mean);
    REGISTER_PLAN_IN(saved_var);
    Tensor& in0 = this->input(0);
    Tensor tensor_eps = empty({1}, device(in0.device()).dtype(in0.data_type()));
    tensor_eps.fill(eps);
    REGISTER_PLAN_IN_HICETENSOR(tensor_eps);
    REGISTER_PLAN_IN(grad_output);
    REGISTER_PLAN_OUT(grad_input);
    REGISTER_PLAN_OUT(grad_scale);
    REGISTER_PLAN_OUT(grad_bias);
  }

  virtual void evaluate();
  virtual void execute();

private:
  double eps_;
};

} // namespace hice

