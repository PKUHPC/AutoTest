#pragma once

#include <hice/intelligent/plan.h>
#include <hice/util/types.h>

namespace hice {

class HICE_API ReLUPlan : public Plan {
public:
  ReLUPlan(const DLTensor& input, 
          DLTensor& output) {
    REGISTER_PLAN_IN(input);
    REGISTER_PLAN_OUT(output);
  }

  virtual void evaluate();
  virtual void execute();

};

// class HICE_API ReLUGradPlan : public Plan {
// public:
//   ReLUGradPlan(const DLTensor& input,
//                 const DLTensor& grad_output,
//                 DLTensor& grad_input) {
//     REGISTER_PLAN_IN(input);
//     REGISTER_PLAN_IN(grad_output);
//     REGISTER_PLAN_OUT(grad_input);
//   }

//   virtual void evaluate();
//   virtual void execute();

// };

} // namespace hice

