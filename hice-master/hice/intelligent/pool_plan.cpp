#include <hice/tvm/tvm.h>
#include <hice/tvm/operators.h>
#include <hice/util/benchmark.h>

#include <hice/intelligent/pool_plan.h>
#include <hice/nn/pooling.h>

namespace hice {

void AvgPoolPlan::evaluate() {
  HICE_PLAN_EVAL(
    [&](){ pooling_avg_fwd(input(0), kernel_, stride_, padding_, output(0)); },
    [&](){ return pooling_avg_fwd_tvm(input(0), kernel_, stride_, padding_, output(0)); }
  );
}

void AvgPoolPlan::execute() {
  HICE_PLAN_EXEC(
    [&](){ pooling_avg_fwd(input(0), kernel_, stride_, padding_, output(0)); },
    [&](){ pooling_avg_fwd_tvm(input(0), kernel_, stride_, padding_, output(0)); }
  );
}


void AvgPoolGradPlan::evaluate() {
  HICE_PLAN_EVAL(
    [&](){ pooling_avg_bwd(input(0), input(1), input(2), kernel_, stride_, padding_, output(0)); },
    [&](){ return pooling_avg_bwd_tvm(input(0), input(1), input(2), kernel_, stride_, padding_, output(0)); }
  );
}

void AvgPoolGradPlan::execute() {
  HICE_PLAN_EXEC(
    [&](){ pooling_avg_bwd(input(0), input(1), input(2), kernel_, stride_, padding_, output(0)); },
    [&](){ pooling_avg_bwd_tvm(input(0), input(1), input(2), kernel_, stride_, padding_, output(0)); }
  );
}

} // namespace hice