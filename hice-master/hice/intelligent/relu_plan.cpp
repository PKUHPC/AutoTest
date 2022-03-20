#include <hice/tvm/tvm.h>
#include <hice/tvm/operators.h>
#include <hice/util/benchmark.h>
#include <hice/util/types.h>

#include <hice/intelligent/relu_plan.h>
#include <hice/nn/activation.h>                                    

namespace hice {

void ReLUPlan::evaluate() {
  HICE_PLAN_EVAL(
    [&](){ relu_fwd(input(0), output(0)); },
    [&](){ return relu_fwd_tvm(input(0), output(0)); }
  );
}

void ReLUPlan::execute() {
  HICE_PLAN_EXEC(
    [&](){ relu_fwd(input(0), output(0)); },
    [&](){ relu_fwd_tvm(input(0), output(0)); }
  );
}


// void ReLUGradPlan::evaluate() {
//   HICE_PLAN_EVAL(
//     [&](){ relu_bwd(input(0), input(1), output(0)); },
//     [&](){ relu_bwd_tvm(input(0), input(1), output(0)); }
//   );
// }

// void ReLUGradPlan::execute() {
//   HICE_PLAN_EXEC(
//     [&](){ relu_bwd(input(0), input(1), output(0)); },
//     [&](){ relu_bwd_tvm(input(0), input(1), output(0)); }
//   );
// }

} // namespace hice