#pragma once

#include <hice/intelligent/plan.h>
#include <hice/util/types.h>

namespace hice {

class HICE_API ConvPlan : public Plan {
public:
  ConvPlan(const DLTensor& input, 
          const DLTensor& weight, 
          // hice::optional<DLTensor> bias,
          ConstIntArrayRef padding,
          ConstIntArrayRef stride,
          ConstIntArrayRef dilation,
          DLTensor& output): 
      stride_(stride),
      padding_(padding), 
      dilation_(dilation) {
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_IN(weight);
    // if (bias) {
    //   REGISTER_PLAN_IN(bias.value());
    // }
    REGISTER_PLAN_OUT(output);
  }

  virtual void evaluate();
  virtual void execute();

private:
  ConstIntArrayRef stride_;
  ConstIntArrayRef padding_;
  ConstIntArrayRef dilation_;
};

class HICE_API ConvGradPlan : public Plan {
public:
  ConvGradPlan(const DLTensor& input, const DLTensor& weight,
              const DLTensor& grad_output,
              ConstIntArrayRef padding, ConstIntArrayRef stride, 
              ConstIntArrayRef dilation,
              DLTensor& grad_input, DLTensor& grad_weight): 
      stride_(stride),
      padding_(padding),
      dilation_(dilation){
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_IN(weight);
    REGISTER_PLAN_IN(grad_output);
    REGISTER_PLAN_OUT(grad_input);
    REGISTER_PLAN_OUT(grad_weight);
  }

  virtual void evaluate();
  virtual void execute();

private:
  ConstIntArrayRef stride_;
  ConstIntArrayRef padding_;
  ConstIntArrayRef dilation_;
};

/// NOTE: If not require to compute grad_input/grad_weight,
/// a empty DLTensor with ndim=0 or data=nullptr should be
/// passed in.
class HICE_API ConvIndependentGradPlan : public Plan {
public:
  ConvIndependentGradPlan(const DLTensor& input, const DLTensor& weight,
              const DLTensor& grad_output,
              ConstIntArrayRef padding, ConstIntArrayRef stride, 
              ConstIntArrayRef dilation,
              DLTensor& grad_input, DLTensor& grad_weight): 
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      idx_grad_input_(-1),
      idx_grad_weight_(-1) {
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_IN(weight);
    REGISTER_PLAN_IN(grad_output);
    if (grad_input.ndim && grad_input.data) {
      REGISTER_PLAN_OUT(grad_input);
      idx_grad_input_ = 0;
    }
    if (grad_weight.ndim && grad_weight.data) {
      REGISTER_PLAN_OUT(grad_weight);
      // might be 0 or 1
      idx_grad_weight_ = outputs().size() - 1;  
    }
  }

  virtual void evaluate();
  virtual void execute();

private:
  ConstIntArrayRef stride_;
  ConstIntArrayRef padding_;
  ConstIntArrayRef dilation_;
  int idx_grad_input_;
  int idx_grad_weight_;
};

} // namespace hice

