#include <hice/tvm/tvm.h>
#include <hice/tvm/operators.h>
#include <hice/util/benchmark.h>

#include <hice/intelligent/batch_norm_plan.h>
#include <hice/nn/batch_norm.h>

namespace hice {

void BatchNormPlan::evaluate() {
  HICE_PLAN_EVAL(
    [&](){ batch_norm_fwd(input(0), input(1), input(2), input(3), input(4), 
                          true, HICE_BATCHNORM_SPATIAL, momentum_, eps_, output(0), output(1), output(2)); },
    [&](){ return batch_norm_fwd_tvm(input(0), input(1), input(2), input(3), input(4),
                              input(5), input(6), 
                              output(0), output(1), output(2)); }
  );
}

void BatchNormPlan::execute() {
  HICE_PLAN_EXEC(
    [&](){ batch_norm_fwd(input(0), input(1), input(2), input(3), input(4), 
                          true, HICE_BATCHNORM_SPATIAL, momentum_, eps_, output(0), output(1), output(2)); },
    [&](){ batch_norm_fwd_tvm(input(0), input(1), input(2), input(3), input(4),
                              input(5), input(6), 
                              output(0), output(1), output(2)); }
  );
}

void BatchNormGradPlan::evaluate() {
  HICE_PLAN_EVAL(
    [&](){ batch_norm_bwd(input(0), input(6), input(1), input(2), input(3), input(4),
                          HICE_BATCHNORM_SPATIAL, eps_,
                          output(0), output(1), output(2)); },
    [&](){ return batch_norm_bwd_tvm(input(0), input(1), input(3), input(4), input(5), input(6),
                              output(0), output(1), output(2)); }
  );
}

void BatchNormGradPlan::execute() {
  HICE_PLAN_EXEC(
    [&](){ batch_norm_bwd(input(0), input(6), input(1), input(2), input(3), input(4),
                          HICE_BATCHNORM_SPATIAL, eps_,
                          output(0), output(1), output(2)); },
    [&](){ batch_norm_bwd_tvm(input(0), input(1), input(3), input(4), input(5), input(6),
                              output(0), output(1), output(2)); }
  );
}

} // namespace hice