#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/core/tensor.h"
#include "hice/math/binary_expr.h"
#include "hice/math/unary_expr.h"
#include "hice/nn/batch_norm.h"

#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

/*
  batch norm case is performed according to the following params:
    1. dims: (n,c,h,w) or (n,c,h,w,d)
    2. bn_mode: spatial, per_activation
    3. forward_mode: training, inference
    4. backward
    5. exponential_factor: (floating number)
    6. epsilon: (floating number for numerical stability)
    Batch normalization forward {4d, spatial, training, exp = 0.5}.
  
  !!! This test code performs dimension check.
  Numerical checks make sure cpu and cuda results correspond.
*/

using namespace hice;
namespace hice {

// size_t getIndex(std::vector<int> &idx, IntArray stride){
//   size_t result = 0;
//   for (int i = 0 ;i < idx.size();++i){
//     result += idx[i]*stride[i];
//   }
//   return result;
// }


void initTensor(Tensor & tensor){
    auto tensor_device = tensor.device();
    auto tensor_scalartype = tensor.scalar_type();
    tensor = tensor.to(kCPU);
    
    if (tensor_scalartype==kFloat){
      float *ptr = tensor.mutable_data<float>();
      for(int i = 0 ; i < tensor.size();++i){
        ptr[i] = i;
      }
    }else if(tensor_scalartype==kDouble){
      double *ptr = tensor.mutable_data<double>();
      for(int i = 0 ; i < tensor.size();++i){
        ptr[i] = i;
      }
    }
    tensor = tensor.to(tensor_device);
}

template <typename TScalarType>
struct BatchNormFwdTestParams {
  ConstIntArrayRef dims;
  int bn_mode;
  bool is_training;
  double expo;
  double epsilon;
  void (* init_function_ptr )(Tensor & );
  hice::Device device;
  hice::ScalarType float_type;
  hice::Device test_device;
};

template <typename TScalarType>
class BatchNormFwdTest
    : public ::testing::TestWithParam<BatchNormFwdTestParams<TScalarType>> {};

using BatchNormFwdTestParamsFloat = BatchNormFwdTestParams<float>;
using BatchNormFwdTestFloat = BatchNormFwdTest<float>;

TEST_P(BatchNormFwdTestFloat, BatchNormFwdTestFloat) {

  // Initializing the test
  TensorPrinter tp;
  BatchNormFwdTestParamsFloat params =
      ::testing::TestWithParam<BatchNormFwdTestParamsFloat>::GetParam();
  auto float_type = params.float_type;
  Tensor test_tensor = full(params.dims, 0, device(params.device).dtype(float_type));

  (*params.init_function_ptr)(test_tensor);
  // tp.print(test_tensor);

  std::vector<int64_t> norm_dims(4, 1);
  if(params.bn_mode == HICE_BATCHNORM_SPATIAL){
    norm_dims[1] = params.dims[1];
    if(params.dims.size() != 4){
      norm_dims.push_back(1);
    }
  }else if(params.bn_mode == HICE_BATCHNORM_PER_ACTIVATION){
    norm_dims[1] = params.dims[1];
    norm_dims[2] = params.dims[2];
    norm_dims[3] = params.dims[3];
    if(params.dims.size() != 4){
     norm_dims.push_back(params.dims[4]);
    }
  }

  Tensor bn_scale = full(norm_dims, 1, device(params.device).dtype(float_type));
  Tensor bn_bias = full(norm_dims, 0, device(params.device).dtype(float_type));
  Tensor running_mean = full(norm_dims, 0, device(params.device).dtype(float_type));
  Tensor running_var = full(norm_dims, 0, device(params.device).dtype(float_type));

  // Perform batch_norm
  auto result = batch_norm_fwd(test_tensor, bn_scale, bn_bias, running_mean,running_var, 
                          params.is_training , params.bn_mode , params.expo,params.epsilon);
  Tensor output = std::get<0>(result);
  Tensor saved_mean = std::get<1>(result);
  Tensor saved_inv_var = std::get<2>(result);

  //Numerical check
  test_tensor = test_tensor.to(params.test_device);
  bn_scale = bn_scale.to(params.test_device);
  bn_bias = bn_bias.to(params.test_device);
  running_mean = running_mean.to(params.test_device);
  running_var = running_var.to(params.test_device);
  
  auto test_result = batch_norm_fwd(test_tensor, bn_scale, bn_bias, running_mean,running_var, 
                          params.is_training , params.bn_mode , params.expo,params.epsilon);
  Tensor test_output = std::get<0>(test_result);  
  
  // Test
  for(int i = 0; i < params.dims.size(); ++i) {
    EXPECT_FLOAT_EQ(output.dims()[i], params.dims[i])<<"Output dimension error. \n"
      <<"Dimension "<<i<<" of output should be "<<params.dims[i]<<" but got "
      <<output.dims()[i]<<". \n";
  }
  for(int i = 0; i < params.dims.size(); ++i) {
    EXPECT_FLOAT_EQ(saved_mean.dims()[i], (norm_dims)[i])<<"Saved_mean dimension error. \n"
      <<"Dimension "<<i<<" of output should be "<<(norm_dims)[i]<<" but got "
      <<output.dims()[i]<<". \n";
  }
  for(int i = 0; i < params.dims.size(); ++i) {
    EXPECT_FLOAT_EQ(saved_inv_var.dims()[i], (norm_dims)[i])<<"Saved_inverse_variance dimension error. \n"
      <<"Dimension "<<i<<" of output should be "<<(norm_dims)[i]<<" but got "
      <<output.dims()[i]<<". \n";
  }
  /*
  auto out_ptr = output.data<float>();
  auto test_out_ptr = test_output.data<float>();
  for(int i = 0 ; i< output.size();++i){
    std::cout<<"DEBUG: "<<i<<std::endl;
    EXPECT_TRUE(-1e-8 < out_ptr[i]-test_out_ptr[i] && 
                out_ptr[i]-test_out_ptr[i] <1e-8)<<"Got inaccurate result"
                <<" at pos "<<i<<" \noutput[i] :"<<out_ptr[i]<<"\n test_output[i]:"<<test_out_ptr[i]<<'\n';
  }*/

  // std::cout<<"\n Params:\n"
  //     <<"device: "<<((params.device==kCUDA)?"GPU":"CPU") << "\nbn_mode: "
  //     <<( (params.bn_mode == HICE_BATCHNORM_PER_ACTIVATION) ? "Per-activation" : "Spatial")<<'\n'
  //     <<"function_mode: " << ((params.is_training )?"Training":"Inference")<<'\n'
  //     <<"{expo_factor, epsilon} = "<<params.expo<<' '<<params.epsilon<<std::endl;

  // std::cout<<"\n--output: "<<std::endl;
  // tp.print(output);
  // std::cout<<"\n--saved mean: "<<std::endl;
  // tp.print(saved_mean);
  // std::cout<<"\n--saved_inv_var: "<<std::endl;
  // tp.print(saved_inv_var);
  // std::cout<<"\n--running mean: "<<std::endl;
  // tp.print(running_mean);
  // std::cout<<"\n--running var: "<<std::endl;
  // tp.print(running_var);
}


// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    BatchNormFwdTestFloatSuite, BatchNormFwdTestFloat,
    ::testing::Values(
      BatchNormFwdTestParamsFloat{{2, 3, 2, 1}, HICE_BATCHNORM_SPATIAL , true, 1, 1e-5, initTensor, kCUDA,kFloat,kCPU},
      BatchNormFwdTestParamsFloat{{2, 3, 2, 1}, HICE_BATCHNORM_PER_ACTIVATION , true, 1, 1e-5, initTensor, kCUDA,kFloat,kCPU},
      BatchNormFwdTestParamsFloat{{2, 3, 2, 1}, HICE_BATCHNORM_SPATIAL , true, 1, 1e-5, initTensor, kCPU,kFloat,kCUDA},
      BatchNormFwdTestParamsFloat{{2, 3, 2, 1}, HICE_BATCHNORM_PER_ACTIVATION , true, 1, 1e-5, initTensor, kCPU,kFloat,kCUDA}
    )
);

} // namespace hice



