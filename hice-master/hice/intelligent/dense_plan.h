#pragma once

#include <hice/intelligent/plan.h>
#include <hice/util/types.h>

namespace hice {

class HICE_API DensePlan : public Plan {
public:
  DensePlan(const DLTensor& input, 
          const DLTensor& weight,
          const DLTensor& bias,
          DLTensor& output) {
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_IN(weight);
    REGISTER_PLAN_IN(bias);
    REGISTER_PLAN_OUT(output);
  }

  virtual void evaluate();
  virtual void execute();

};

class HICE_API DenseGradPlan : public Plan {
public:
  DenseGradPlan(const DLTensor& input, const DLTensor& weight,
                const DLTensor& grad_output,
                DLTensor& grad_input,
                DLTensor& grad_weight,
                DLTensor& grad_bias) {
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_IN(weight);
    REGISTER_PLAN_IN(grad_output);
    REGISTER_PLAN_OUT(grad_input);
    REGISTER_PLAN_OUT(grad_weight);
    REGISTER_PLAN_OUT(grad_bias);
  }

  virtual void evaluate();
  virtual void execute();

};

} // namespace hice

