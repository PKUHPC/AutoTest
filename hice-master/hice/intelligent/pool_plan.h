#pragma once

#include <hice/intelligent/plan.h>
#include <hice/util/types.h>

namespace hice {

class HICE_API AvgPoolPlan : public Plan {
public:
  AvgPoolPlan(const DLTensor& input, 
              ConstIntArrayRef kernel,
              ConstIntArrayRef stride,
              ConstIntArrayRef padding,
              DLTensor& output): 
      kernel_(kernel),
      stride_(stride),
      padding_(padding) {
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_OUT(output);
  }

  virtual void evaluate();
  virtual void execute();

private:
  ConstIntArrayRef kernel_;
  ConstIntArrayRef stride_;
  ConstIntArrayRef padding_;
};

class HICE_API AvgPoolGradPlan : public Plan {
public:
  AvgPoolGradPlan(const DLTensor& input, const DLTensor& output,
                  const DLTensor& grad_output,
                  ConstIntArrayRef kernel, ConstIntArrayRef stride,
                  ConstIntArrayRef padding, DLTensor& grad_input): 
      kernel_(kernel),
      stride_(stride),
      padding_(padding) {
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_IN(output);
    REGISTER_PLAN_IN(grad_output);
    REGISTER_PLAN_OUT(grad_input);
  }

  virtual void evaluate();
  virtual void execute();

private:
  ConstIntArrayRef kernel_;
  ConstIntArrayRef stride_;
  ConstIntArrayRef padding_;
};

} // namespace hice

