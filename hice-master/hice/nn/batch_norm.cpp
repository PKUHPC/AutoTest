#include <iostream>
#include <vector>

#include "hice/basic/factories.h"
#include "hice/nn/batch_norm.h"

namespace hice {

HICE_DEFINE_DISPATCHER(batch_norm_fwd_dispatcher);
HICE_DEFINE_DISPATCHER(batch_norm_bwd_dispatcher);

std::tuple<Tensor, Tensor, Tensor> batch_norm_fwd(
    Tensor &input, Tensor &bn_scale, Tensor &bn_bias, Tensor &running_mean,
    Tensor &running_var, bool training, uint32_t mode,
    double exponential_average_factor, double epsilon) {
  /**
   * This function implements batch normalization for hice.
   * It takes input params as follows:
   *  Tensor & input: 4d or 5d tensor, a batch of incoming data
   *  Tensor & bn_scale: original scaling factor
   *  Tensor & bn_bias: original bias
   *  Tensor & running_mean: used to estimated the mean of the whole training
   *set. (Should be carefully initiated e.g., to 0 or to the mean of the first
   *batch) Tensor & running_var: similar as running_mean above. bool train: to
   *support training mode and inference mode int mode: mode == 0 for
   *per-activation batch-norm; mode== 1 for spatial batch-norm in per-activation
   *mode, input(N * C * H * W) -> saved mean(1 * C * H * W) in spatial mode,
   *input(N * C * H * W) -> saved mean(1 * C * 1 * 1) double epsilon: for
   *numerical satbility double expo_factor: the factor used to calculated
   *running_{mean,var} running_mean_updated = (1-expo_factor) * running_mean +
   *expo_factor * new_batch_mean. Similar for running_var.
   *
   * It return std::tuple as output and its members are:
   *  Tensor output: input after being normalized -- output = (input-
   *input.mean) / sqrt(epsilon + input.var) Tensor saved_mean: The mean of the
   *incoming batch. For backpropagation. (makes no sense in inference mode)
   *  Tensor saved_inv_var: The INVERSER of the square of variance of the
   *incoming batch, that is,  1 / sqrt(epsilon + input.var) (makes no sense in
   *inference mode)
   **/

  auto input_shape = input.dims();
  auto input_device = input.device();
  std::vector<int64_t> bn_shape_vec;

  // 1st dimention is 1
  bn_shape_vec.push_back(1);

  Tensor output = full(input_shape, 0., device(input_device).dtype(input.scalar_type()));

  if(mode == HICE_BATCHNORM_PER_ACTIVATION){//per-activation
    for(int i = 1 ; i < input_shape.size(); ++i){
      bn_shape_vec.push_back(input_shape[i]);
    }
  }else if(mode == HICE_BATCHNORM_SPATIAL){//spatial
    bn_shape_vec.push_back(input_shape[1]);
    for(int i = 2; i < input_shape.size(); ++i){
      bn_shape_vec.push_back(1);
    }
  }else{
    std::cout<<"error in mode."<<std::endl;
    exit(0);
  }

  IntArrayRef bn_shape(bn_shape_vec);
  Tensor saved_mean = full(bn_shape, 0., device(input_device).dtype(input.scalar_type()));
  Tensor saved_inv_var = full(bn_shape, 0., device(input_device).dtype(input.scalar_type()));
  batch_norm_fwd_dispatcher(
      input, output, bn_scale, bn_bias, running_mean, running_var, training,
      mode, epsilon, exponential_average_factor, saved_mean, saved_inv_var);

  return std::make_tuple(output, saved_mean, saved_inv_var);
}


std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_fwd(
    Tensor &input, Tensor &bn_scale, Tensor &bn_bias, Tensor &running_mean,
    Tensor &running_var, bool training, uint32_t mode,
    double exponential_average_factor, double epsilon, Tensor& output, Tensor& saved_mean, Tensor& saved_inv_var) {

  batch_norm_fwd_dispatcher(
      input, output, bn_scale, bn_bias, running_mean, running_var, training,
      mode, epsilon, exponential_average_factor, saved_mean, saved_inv_var);

  return std::tuple<Tensor&, Tensor&, Tensor&>{output, saved_mean, saved_inv_var};
}


std::tuple<Tensor,Tensor,Tensor> batch_norm_bwd(
  Tensor &input, Tensor &output_grad,
  Tensor & bn_scale, Tensor & bn_bias,
  Tensor & saved_mean, Tensor & saved_inv_var,
  uint32_t mode, double epsilon){
/**
 * This function implements batch normalization for hice.
 * It takes input params as follows:
 *  Tensor & input: 4d or 5d tensor, a batch of incoming data
 *  Tensor & output_grad: grad from upstream.
 *  Tensor & bn_scale: original scaling factor
 *  Tensor & bn_bias: original bias
 *  Tensor saved_mean: The mean of the incoming batch. For backpropagation.
 *  Tensor saved_inv_var: The INVERSER of the variance of the incoming batch.
 *  int mode: mode ==0 for per-activation batch-norm; mode==1 for spatial batch-norm
 *            in per-activation mode, input(N * C * H * W) -> saved mean(1 * C * H * W)
 *            in spatial mode, input(N * C * H * W) -> saved mean(1 * C * 1 * 1)
 *  double epsilon: for numerical satbility
 * 
 * It return std::tuple as output and its members are:
 *  Tensor input_grad
 *  Tensor bn_scale_grad
 *  Tensor bn_bias_grad
 * 
 **/

  auto input_device = input.device();
  Tensor bn_scale_grad = full(bn_scale.dims(), 0, device(input_device).dtype(input.scalar_type()));
  Tensor bn_bias_grad = full(bn_bias.dims(),0, device(input_device).dtype(input.scalar_type()));
  Tensor input_grad = full(input.dims(), 0, device(input_device).dtype(input.scalar_type()));

  batch_norm_bwd_dispatcher(input, output_grad, bn_scale, bn_bias, saved_mean,
                            saved_inv_var, mode, epsilon, bn_scale_grad,
                            bn_bias_grad, input_grad);

  return std::make_tuple(input_grad, bn_scale_grad, bn_bias_grad);
}


std::tuple<Tensor&,Tensor&,Tensor&> batch_norm_bwd(
  Tensor &input, Tensor &output_grad,
  Tensor & bn_scale, Tensor & bn_bias,
  Tensor & saved_mean, Tensor & saved_inv_var,
  uint32_t mode, double epsilon, Tensor& input_grad, Tensor& bn_scale_grad, Tensor& bn_bias_grad){
  
  batch_norm_bwd_dispatcher(input, output_grad, bn_scale, bn_bias, saved_mean,
                            saved_inv_var, mode, epsilon, bn_scale_grad,
                            bn_bias_grad, input_grad);

  return std::tuple<Tensor&, Tensor&, Tensor&>{input_grad, bn_scale_grad, bn_bias_grad};
}


} // namespace hice
