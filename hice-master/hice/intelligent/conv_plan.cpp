#include <hice/tvm/tvm.h>
#include <hice/tvm/operators.h>
#include <hice/util/benchmark.h>
#include <hice/util/types.h>

#include <hice/intelligent/conv_plan.h>
#include <hice/nn/conv.h>                                    

namespace hice {

void ConvPlan::evaluate() {
  HICE_PLAN_EVAL(
    [&](){ conv_fwd(input(0), input(1), hice::nullopt, padding_, stride_, dilation_, 1, false, true, output(0)); },
    [&](){ return conv_fwd_tvm(input(0), input(1), padding_, stride_, dilation_, output(0)); }
  );
}

void ConvPlan::execute() {
  HICE_PLAN_EXEC(
    [&](){ conv_fwd(input(0), input(1), hice::nullopt, padding_, stride_, dilation_, 1, false, true, output(0)); },
    [&](){ conv_fwd_tvm(input(0), input(1), padding_, stride_, dilation_, output(0)); }
  );
}


void ConvGradPlan::evaluate() {
  HICE_PLAN_EVAL(
    [&](){ conv_bwd(input(0), input(1), input(2), padding_, stride_, dilation_, 1, false, true, output(0), output(1), hice::nullopt); },
    [&](){ return conv_bwd_tvm(input(0), input(1), input(2), padding_, stride_, dilation_, output(0), output(1)); }
  );
}

void ConvGradPlan::execute() {
  HICE_PLAN_EXEC(
    [&](){ conv_bwd(input(0), input(1), input(2), padding_, stride_, dilation_, 1, false, true, output(0), output(1), hice::nullopt); },
    [&](){ conv_bwd_tvm(input(0), input(1), input(2), padding_, stride_, dilation_, output(0), output(1)); }
  );
}


void ConvIndependentGradPlan::evaluate() {
  HICE_PLAN_EVAL(
    [&](){
      hice::optional<hice::Tensor> grad_input;
      hice::optional<hice::Tensor> grad_weight;
      if (idx_grad_input_ != -1) {
        grad_input = output(idx_grad_input_);
      }
      if (idx_grad_weight_ != -1) {
        grad_weight = output(idx_grad_weight_);
      }
      conv_bwd(input(0), input(1), input(2), padding_, stride_, dilation_, 1, false, true, grad_input, grad_weight, hice::nullopt); 
    },
    [&](){
      bool status = true;
      if (idx_grad_input_ != -1) {
        status = status && conv_bwd_input_tvm(
          input(0), input(1), input(2), padding_, stride_, dilation_, output(idx_grad_input_));
      }
      if (idx_grad_weight_ != -1) {
        status = status && conv_bwd_weight_tvm(
          input(0), input(1), input(2), padding_, stride_, dilation_, output(idx_grad_weight_));
      }
      return status;
    }
  );
}

void ConvIndependentGradPlan::execute() {
  HICE_PLAN_EXEC(
    [&](){ 
      hice::optional<hice::Tensor> grad_input;
      hice::optional<hice::Tensor> grad_weight;
      if (idx_grad_input_ != -1) {
        grad_input = output(idx_grad_input_);
      }
      if (idx_grad_weight_ != -1) {
        grad_weight = output(idx_grad_weight_);
      }
      conv_bwd(input(0), input(1), input(2), padding_, stride_, dilation_, 1, false, true, grad_input, grad_weight, hice::nullopt); 
    },
    [&](){
      if (idx_grad_input_ != -1) {
        conv_bwd_input_tvm(input(0), input(1), input(2), padding_, stride_, dilation_, output(idx_grad_input_));
      }
      if (idx_grad_weight_ != -1) {
        conv_bwd_weight_tvm(input(0), input(1), input(2), padding_, stride_, dilation_, output(idx_grad_weight_));
      }
    }
  );
}

} // namespace hice