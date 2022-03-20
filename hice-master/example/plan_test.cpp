#include <hice/nn/conv.h>
#include <hice/intelligent/conv_plan.h>

#include <hice/nn/activation.h>
#include <hice/intelligent/relu_plan.h>

#include <hice/nn/batch_norm.h>
#include <hice/intelligent/batch_norm_plan.h>

#include <hice/nn/pooling.h>
#include <hice/intelligent/pool_plan.h>

#include <hice/intelligent/dense_plan.h>

#include <hice/util/types.h>
#include <hice/core/tensor_printer.h>
#include <hice/tvm/tvm.h>

#include <cmath>

using namespace hice;

inline void ExpectEqualDenseRegardlessDevice(const hice::Tensor& tensor1,
                                             const hice::Tensor& tensor2,
                                            float allowed_err=1e-5) {
  HICE_CHECK_EQ(tensor1.size(), tensor2.size());
  HICE_CHECK_EQ(tensor1.offset(), tensor2.offset());
  HICE_CHECK_EQ(tensor1.data_type(), tensor2.data_type());
  HICE_CHECK_EQ(tensor1.ndim(), tensor2.ndim());
  HICE_CHECK_EQ(tensor1.shape(), tensor2.shape());
  HICE_CHECK_EQ(tensor1.strides(), tensor2.strides());
  hice::Tensor tensor1_new =
      tensor1.device_type() == kCPU ? tensor1 : tensor1.to(kCPU);
  hice::Tensor tensor2_new =
      tensor2.device_type() == kCPU ? tensor2 : tensor2.to(kCPU);
  auto size = tensor1.size();
  auto sc_type = tensor1.scalar_type();
  HICE_DISPATCH_FLOATING_TYPES(sc_type, "", [&]() {
    double err_max = 0, err = 0;
    for (int i = 0; i < size; ++i) {
      err = std::abs(tensor1_new.data<scalar_t>()[i] - tensor2_new.data<scalar_t>()[i]);
      err_max = std::max(err_max, err);
    }
    bool passed = err_max < allowed_err;
    if (!passed) {
      std::cout<<"err_max = "<<err_max<<std::endl;
    }
    HICE_CHECK(passed);
  });
}

void pool_test() {
  float one_val = 1.0, zero_val = 0.0;
  float mean_val = 0.0, std_val = 1.0;

  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> padding = {0, 0};
  std::vector<int64_t> kernel = {7, 7};
  std::vector<int64_t> dims_in = {10, 10, 10, 10};
  std::vector<int64_t> dims_out = {10, 10, 4, 4};

  hice::Tensor input = rand_normal(dims_in, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor output = empty(dims_out, device(kCUDA).dtype(kFloat));
  DLTensor dl_input = HICETensor_to_DLTensor(input);
  DLTensor dl_output = HICETensor_to_DLTensor(output);
  PlanPtr plan_ptr;
  plan_ptr = make_plan<AvgPoolPlan>(dl_input, kernel, stride, padding, dl_output);
  plan_ptr->evaluate();
  HICE_DLOG(INFO) << "avg_pool_plan evaluate end.";
  plan_ptr->execute();
  
  hice::Tensor output_hice = empty(dims_out, device(kCUDA).dtype(kFloat));
  hice::pooling_avg_fwd(input, kernel, stride, padding, output_hice);
  ExpectEqualDenseRegardlessDevice(output, output_hice);

  hice::Tensor grad_output = rand_normal(dims_out, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor grad_input = empty_like(input);
  DLTensor dl_grad_input = HICETensor_to_DLTensor(grad_input);
  DLTensor dl_grad_output = HICETensor_to_DLTensor(grad_output);
  plan_ptr->update_input_dataptr(0, dl_input.data);
  plan_ptr = make_plan<AvgPoolGradPlan>(dl_input, dl_output, dl_grad_output, kernel, stride, padding, dl_grad_input);
  plan_ptr->evaluate();
  HICE_DLOG(INFO) << "avg_pool_grad_plan evaluate end.";
  plan_ptr->execute();
  
  hice::Tensor grad_input_hice = empty(dims_in, device(kCUDA).dtype(kFloat));
  hice::pooling_avg_bwd(input, output, grad_output, kernel, stride, padding, grad_input_hice);
  ExpectEqualDenseRegardlessDevice(grad_input, grad_input_hice);
}

void conv_test() {
  float one_val = 1.0, zero_val = 0.0;
  float mean_val = 0.0, std_val = 1.0;
  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> padding = {1, 1};
  std::vector<int64_t> dilation = {1, 1};
  std::vector<int64_t> dims_in = {32, 3, 20, 20};
  std::vector<int64_t> dims_weight = {32, 3, 3, 3};
  std::vector<int64_t> dims_out = {32, 32, 20, 20};
  
  hice::Tensor input = rand_normal(dims_in, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor weight = rand_normal(dims_weight, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor output = empty(dims_out, device(kCUDA).dtype(kFloat));
  DLTensor dl_input = HICETensor_to_DLTensor(input);
  DLTensor dl_weight = HICETensor_to_DLTensor(weight);
  DLTensor dl_output = HICETensor_to_DLTensor(output);
  PlanPtr plan_ptr;
  plan_ptr = make_plan<ConvPlan>(dl_input, dl_weight, padding, stride, dilation, dl_output);
  plan_ptr->evaluate();
  HICE_DLOG(INFO) << "ConvPlan evaluate end.";
  plan_ptr->execute();
  plan_ptr->set_impl_type(ImplementationType::kOfficial);
  plan_ptr->execute();

  hice::Tensor output_hice = empty(dims_out, device(kCUDA).dtype(kFloat));
  hice::conv_fwd(input, weight, hice::nullopt, padding, stride, dilation, 1, false, true, output_hice);
  ExpectEqualDenseRegardlessDevice(output, output_hice, 1e-3);

  hice::Tensor grad_output = rand_normal(dims_out, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor grad_input = empty_like(input);
  hice::Tensor grad_weight = empty_like(weight);
  DLTensor dl_grad_output = HICETensor_to_DLTensor(grad_output);
  DLTensor dl_grad_input = HICETensor_to_DLTensor(grad_input);
  DLTensor dl_grad_weight = HICETensor_to_DLTensor(grad_weight);
  // DLTensor dl_grad_input{nullptr, DLContext(), 0, {0, 0, 0}, nullptr, nullptr, 0};
  // DLTensor dl_grad_weight{nullptr, DLContext(), 0, {0, 0, 0}, nullptr, nullptr, 0};
  plan_ptr = make_plan<ConvGradPlan>(dl_input, dl_weight, dl_grad_output, padding, stride, dilation, dl_grad_input, dl_grad_weight);
  // plan_ptr->evaluate();
  HICE_DLOG(INFO) << "ConvGradPlan evaluate end.";
  plan_ptr->execute();

  hice::Tensor grad_input_hice = empty_like(input);
  hice::Tensor grad_weight_hice = empty_like(weight);
  hice::conv_bwd(input, weight, grad_output, padding, stride, dilation, 
    1, false, true, grad_input_hice, grad_weight_hice, hice::nullopt);
  ExpectEqualDenseRegardlessDevice(grad_input, grad_input_hice, 1e-3);
  ExpectEqualDenseRegardlessDevice(grad_weight, grad_weight_hice, 1e-3);
}

void dense_test() {
  float one_val = 1.0, zero_val = 0.0;
  float mean_val = 0.0, std_val = 1.0;
  int N = 10, CI = 6, CO = 6;
  std::vector<int64_t> dims_in = {N, CI};
  std::vector<int64_t> dims_weight = {CO, CI};
  std::vector<int64_t> dims_out = {N, CO};
  std::vector<int64_t> dims_bias = {CO};
  hice::Tensor input = rand_normal(dims_in, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor weight = rand_normal(dims_weight, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor bias = rand_normal(dims_bias, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor output = empty(dims_out, device(kCUDA).dtype(kFloat));
  DLTensor dl_input = HICETensor_to_DLTensor(input);
  DLTensor dl_weight = HICETensor_to_DLTensor(weight);
  DLTensor dl_bias = HICETensor_to_DLTensor(bias);
  DLTensor dl_output = HICETensor_to_DLTensor(output);
  PlanPtr plan_ptr;
  plan_ptr = make_plan<DensePlan>(dl_input, dl_weight, dl_bias, dl_output);
  plan_ptr->evaluate();
  HICE_DLOG(INFO) << "DensePlan evaluate end.";
  plan_ptr->execute();

  // hice::Tensor output_hice = empty(dims_out, device(kCUDA).dtype(kFloat));
  // hice::conv_fwd(input, weight, hice::nullopt, padding, stride, dilation, 1, false, true, output_hice);
  // ExpectEqualDenseRegardlessDevice(output, output_hice);

  hice::Tensor grad_output = rand_normal(dims_out, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor grad_input = empty_like(input);
  hice::Tensor grad_weight = empty_like(weight);
  hice::Tensor grad_bias = empty_like(bias);
  DLTensor dl_grad_output = HICETensor_to_DLTensor(grad_output);
  DLTensor dl_grad_input = HICETensor_to_DLTensor(grad_input);
  DLTensor dl_grad_weight = HICETensor_to_DLTensor(grad_weight);
  DLTensor dl_grad_bias = HICETensor_to_DLTensor(grad_bias);
  plan_ptr = make_plan<DenseGradPlan>(dl_input, dl_weight, dl_grad_output, dl_grad_input, dl_grad_weight, dl_grad_bias);
  plan_ptr->evaluate();
  HICE_DLOG(INFO) << "DenseGradPlan evaluate end.";
  plan_ptr->execute();
}

void batch_norm_test() {
  float one_val = 1.0, zero_val = 0.0;
  float mean_val = 0.0, std_val = 1.0;
  int64_t N = 10, C = 6, H = 5, W = 5;
  double eps = 1e-5, momentum = 0.1;
  std::vector<int64_t> dims = {N, C, H, W};
  std::vector<int64_t> dims_ch = {C};
  
  hice::Tensor input = rand_normal(dims, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor scale = rand_normal(dims_ch, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor bias = rand_normal(dims_ch, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor running_mean = full(dims_ch, zero_val, device(kCUDA).dtype(kFloat));
  hice::Tensor running_var = full(dims_ch, one_val, device(kCUDA).dtype(kFloat));
  hice::Tensor output = empty(dims, device(kCUDA).dtype(kFloat));
  hice::Tensor saved_mean = empty(dims_ch, device(kCUDA).dtype(kFloat));
  hice::Tensor saved_var = empty(dims_ch, device(kCUDA).dtype(kFloat));
  DLTensor dl_input = HICETensor_to_DLTensor(input);
  DLTensor dl_scale = HICETensor_to_DLTensor(scale);
  DLTensor dl_bias = HICETensor_to_DLTensor(bias);
  DLTensor dl_running_mean = HICETensor_to_DLTensor(running_mean);
  DLTensor dl_running_var = HICETensor_to_DLTensor(running_var);
  DLTensor dl_output = HICETensor_to_DLTensor(output);
  DLTensor dl_saved_mean = HICETensor_to_DLTensor(saved_mean);
  DLTensor dl_saved_var = HICETensor_to_DLTensor(saved_var);

  PlanPtr plan_ptr;
  plan_ptr = make_plan<BatchNormPlan>(dl_input, dl_scale, dl_bias, dl_running_mean, dl_running_var, momentum, eps,
                                      dl_output, dl_saved_mean, dl_saved_var);
  plan_ptr->evaluate();
  HICE_DLOG(INFO) << "BatchNormPlan evaluate end.";
  plan_ptr->execute();

  hice::Tensor output_hice = empty(dims, device(kCUDA).dtype(kFloat));
  hice::Tensor saved_mean_hice = empty(dims_ch, device(kCUDA).dtype(kFloat));
  hice::Tensor saved_var_hice = empty(dims_ch, device(kCUDA).dtype(kFloat));
  hice::batch_norm_fwd(input, scale, bias, running_mean, running_var, 
                        true, HICE_BATCHNORM_SPATIAL, momentum, eps, 
                        output_hice, saved_mean_hice, saved_var_hice);
  ExpectEqualDenseRegardlessDevice(output, output_hice, 1e-3);

  hice::Tensor grad_output = rand_normal(dims, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor grad_input = empty_like(input);
  hice::Tensor grad_scale = empty_like(scale);
  hice::Tensor grad_bias = empty_like(bias);
  DLTensor dl_grad_output = HICETensor_to_DLTensor(grad_output);
  DLTensor dl_grad_input = HICETensor_to_DLTensor(grad_input);
  DLTensor dl_grad_scale = HICETensor_to_DLTensor(grad_scale);
  DLTensor dl_grad_bias = HICETensor_to_DLTensor(grad_bias);
  plan_ptr = make_plan<BatchNormGradPlan>(dl_input, dl_scale, dl_bias, dl_saved_mean, dl_saved_var, dl_grad_output, eps, 
                          dl_grad_input, dl_grad_scale, dl_grad_bias);
  plan_ptr->evaluate();
  HICE_DLOG(INFO) << "BatchNormGradPlan evaluate end.";
  plan_ptr->execute();

  hice::Tensor grad_input_hice = empty_like(input);
  hice::Tensor grad_scale_hice = empty_like(scale);
  hice::Tensor grad_bias_hice = empty_like(bias);
  hice::batch_norm_bwd(input, grad_output, scale, bias, saved_mean_hice, saved_var_hice, 
                    HICE_BATCHNORM_SPATIAL, eps, 
                    grad_input_hice, grad_scale_hice, grad_bias_hice);
  ExpectEqualDenseRegardlessDevice(grad_input, grad_input_hice, 1e-3);
  ExpectEqualDenseRegardlessDevice(grad_scale, grad_scale_hice, 1e-3);
  ExpectEqualDenseRegardlessDevice(grad_bias, grad_bias_hice, 1e-3);

  // TensorPrinter tp;
  // tp.print(grad_input);
  // tp.print(grad_input_hice);
}

void relu_test() {
  float one_val = 1.0, zero_val = 0.0;
  float mean_val = 0.0, std_val = 1.0;
  int N = 10, C = 10, H = 6, W = 5;
  std::vector<int64_t> dims = {N, C, H, W};
  hice::Tensor input = rand_normal(dims, mean_val, std_val, device(kCUDA).dtype(kFloat));
  hice::Tensor output = empty(dims, device(kCUDA).dtype(kFloat));
  DLTensor dl_input = HICETensor_to_DLTensor(input);
  DLTensor dl_output = HICETensor_to_DLTensor(output);
  PlanPtr plan_ptr;
  plan_ptr = make_plan<ReLUPlan>(dl_input, dl_output);
  plan_ptr->evaluate();
  HICE_DLOG(INFO) << "ReLUPlan evaluate end.";
  plan_ptr->execute();

  hice::Tensor output_hice = empty(dims, device(kCUDA).dtype(kFloat));
  hice::relu_fwd(input, output_hice);
  ExpectEqualDenseRegardlessDevice(output, output_hice);
}

int main() {
  TVMLibConfig::set_prefix("./hice_tmp");
  TVMLibConfig::set_n_search_trails(2);
  loguru::g_stderr_verbosity = loguru::NamedVerbosity::Verbosity_OFF;
  loguru::g_internal_verbosity = loguru::NamedVerbosity::Verbosity_OFF;

  pool_test();
  // conv_test();
  // dense_test();
  // batch_norm_test();
  // relu_test();
  // DLTensor dl_grad_input{nullptr, DLContext(), 0, {0, 0, 0}, nullptr, nullptr, 0};
  // HICE_DLOG(INFO) << dl_input.data;

  // hice::optional<hice::Tensor> self;
  // HICE_DLOG(INFO) << !self;

  return 0;
}