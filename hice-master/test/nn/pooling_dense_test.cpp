#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/nn/pooling.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"

// #if 0

namespace hice {

template <typename TScalarType>
struct PoolingAVG_FWDTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> kernel;
  std::vector<int64_t> strides;
  std::vector<int64_t> padding;
  std::vector<int64_t> dims_output;
};

// avg fwd outplace
template <typename TScalarType>
class PoolingAVG_FWDTest
    : public ::testing::TestWithParam<PoolingAVG_FWDTestParams<TScalarType>> {};

using PoolingAVG_FWDTestParamsFloat = PoolingAVG_FWDTestParams<float>;
using PoolingAVG_FWDTestFloat = PoolingAVG_FWDTest<float>;

TEST_P(PoolingAVG_FWDTestFloat, Pooling_AVG_FWD_OutplaceTestFloat) {
  PoolingAVG_FWDTestParamsFloat params =
      ::testing::TestWithParam<PoolingAVG_FWDTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_output = pooling_avg_fwd(cpu_input, params.kernel, 
                                        params.strides, params.padding);
  // TensorPrinter tp;
  Tensor cuda_input = cpu_input.to(kCUDA);   
  Tensor cuda_output = pooling_avg_fwd(cuda_input, params.kernel, 
                                         params.strides, params.padding);
  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

#if 1
// avg fwd inplace
TEST_P(PoolingAVG_FWDTestFloat, Pooling_AVG_FWD_InplaceTestFloat) {
  PoolingAVG_FWDTestParamsFloat params =
      ::testing::TestWithParam<PoolingAVG_FWDTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_output = empty(params.dims_output, device(kCPU).dtype(kFloat));
  pooling_avg_fwd(cpu_input, params.kernel, 
                  params.strides, params.padding,
                  cpu_output);
  // TensorPrinter tp;
  Tensor cuda_input = cpu_input.to(kCUDA);   
  Tensor cuda_output = empty(params.dims_output, device(kCUDA).dtype(kFloat));
  pooling_avg_fwd(cuda_input, params.kernel, 
                  params.strides, params.padding,
                  cuda_output);
  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}
#endif

INSTANTIATE_TEST_CASE_P(
    PoolingAVG_FWDTestFloatSuite, PoolingAVG_FWDTestFloat,
    ::testing::Values(
      // pooling1d
      PoolingAVG_FWDTestParamsFloat{{5, 3, 5}, // dims of input
                                    {2},       // kernel(H, W)
                                    {1},       // stride(H, W)
                                    {0},       // padding(H, W)
                                    {5, 3, 4}},// dims of output
      PoolingAVG_FWDTestParamsFloat{{5, 3, 4}, 
                                    {2},       
                                    {1},      
                                    {0},     
                                    {5, 3, 3}},
      PoolingAVG_FWDTestParamsFloat{{3, 2, 6}, 
                                    {2},
                                    {2}, 
                                    {1},  
                                    {3, 2, 4}}, 
      PoolingAVG_FWDTestParamsFloat{{4, 2, 6}, 
                                    {2},
                                    {3}, 
                                    {1},  
                                    {4, 2, 3}},
      // pooling2d
      PoolingAVG_FWDTestParamsFloat{{5, 3, 5, 4}, // dims of input
                                    {2, 2},       // kernel(H, W)
                                    {1, 1},       // stride(H, W)
                                    {0, 0},       // padding(H, W)
                                    {5, 3, 4, 3}},// dims of output
      PoolingAVG_FWDTestParamsFloat{{5, 3, 5, 4}, 
                                    {2, 2},       
                                    {1},      
                                    {0},     
                                    {5, 3, 4, 3}},
      PoolingAVG_FWDTestParamsFloat{{3, 2, 4, 6}, 
                                    {3, 2},
                                    {2, 2}, 
                                    {1, 1},  
                                    {3, 2, 2, 4}}, 
      PoolingAVG_FWDTestParamsFloat{{4, 2, 6, 6}, 
                                    {2, 3},
                                    {3, 2}, 
                                    {1, 2},  
                                    {4, 2, 3, 4}},
      // pooling3d
      PoolingAVG_FWDTestParamsFloat{{5, 3, 5, 4, 5},  // dims of input
                                    {2, 2, 2},        // kernel(D, H, W)
                                    {},               // stride(D, H, W)
                                    {},               // padding(D, H, W)
                                    {5, 3, 4, 3, 4}}, // dims of output
      PoolingAVG_FWDTestParamsFloat{{3, 2, 4, 6, 10}, 
                                    {3, 2, 4},
                                    {2, 2, 2}, 
                                    {1, 1, 1},  
                                    {3, 2, 2, 4, 5}}, 
      PoolingAVG_FWDTestParamsFloat{{4, 2, 6, 6, 12}, 
                                    {2, 3, 2},
                                    {3, 2, 4}, 
                                    {1, 2, 1},  
                                    {4, 2, 3, 4, 4}}
    )
);

// #if 0

template <typename TScalarType>
struct PoolingAVG_BWDTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> dims_grad_output;
  std::vector<int64_t> kernel;
  std::vector<int64_t> strides;
  std::vector<int64_t> padding;
  std::vector<int64_t> dims_grad_input;
};

template <typename TScalarType>
class PoolingAVG_BWDTest
    : public ::testing::TestWithParam<PoolingAVG_BWDTestParams<TScalarType>> {};

using PoolingAVG_BWDTestParamsFloat = PoolingAVG_BWDTestParams<float>;
using PoolingAVG_BWDTestFloat = PoolingAVG_BWDTest<float>;

// avg bwd outplace
TEST_P(PoolingAVG_BWDTestFloat, Pooling_AVG_BWD_OutplaceTestFloat) {
  PoolingAVG_BWDTestParamsFloat params =
      ::testing::TestWithParam<PoolingAVG_BWDTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_output = pooling_avg_fwd(cpu_input, params.kernel, 
                                      params.strides, params.padding);
  Tensor cpu_grad_output = rand_uniform(params.dims_grad_output, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cuda_input = cpu_input.to(kCUDA);   
  Tensor cuda_output = cpu_output.to(kCUDA);   
  Tensor cuda_grad_output = cpu_grad_output.to(kCUDA);   
  Tensor cpu_grad_input = pooling_avg_bwd(cpu_input, cpu_output,
                                      cpu_grad_output,
                                      params.kernel, params.strides, 
                                      params.padding);
  // TensorPrinter tp;
  Tensor cuda_grad_input = pooling_avg_bwd(cuda_input, cuda_output,
                                      cuda_grad_output,
                                      params.kernel, params.strides, 
                                      params.padding);
  ExpectEqualDenseRegardlessDevice(cpu_grad_input, cuda_grad_input);
}

#if 1
// avg bwd inplace
TEST_P(PoolingAVG_BWDTestFloat, Pooling_AVG_BWD_InplaceTestFloat) {
  PoolingAVG_BWDTestParamsFloat params =
      ::testing::TestWithParam<PoolingAVG_BWDTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_output = empty(params.dims_grad_output, device(kCPU).dtype(kFloat));
  pooling_avg_fwd(cpu_input, params.kernel, params.strides, params.padding, cpu_output);
  Tensor cpu_grad_output = rand_uniform(params.dims_grad_output, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_grad_input = empty(params.dims_grad_input, device(kCPU).dtype(kFloat));
  Tensor cuda_input = cpu_input.to(kCUDA);   
  Tensor cuda_output = cpu_output.to(kCUDA);   
  Tensor cuda_grad_output = cpu_grad_output.to(kCUDA);   
  Tensor cuda_grad_input = cpu_grad_input.to(kCUDA);   
  pooling_avg_bwd(cpu_input, cpu_output, cpu_grad_output, 
                  params.kernel, params.strides, 
                  params.padding, cpu_grad_input);
  // TensorPrinter tp;
  pooling_avg_bwd(cuda_input, cuda_output, cuda_grad_output,
                  params.kernel, params.strides, 
                  params.padding, cuda_grad_input);
  ExpectEqualDenseRegardlessDevice(cpu_grad_input, cuda_grad_input);
}
#endif

INSTANTIATE_TEST_CASE_P(
    PoolingAVG_BWDTestFloatSuite, PoolingAVG_BWDTestFloat,
    ::testing::Values(
      // pooling1d
      PoolingAVG_BWDTestParamsFloat{{5, 3, 5}, // dims of input
                                    {5, 3, 4},// dims of grad_output
                                    {2},       // kernel(H, W)
                                    {1},       // stride(H, W)
                                    {0},       // padding(H, W)
                                    {5, 3, 5}}// dims of grad_input
      // PoolingAVG_BWDTestParamsFloat{{5, 3, 4}, 
      //                               {5, 3, 3},
      //                               {2},       
      //                               {},       
      //                               {},      
      //                               {5, 3, 4}},
      // PoolingAVG_BWDTestParamsFloat{{5, 3, 5}, 
      //                               {5, 3, 4},
      //                               {2},       
      //                               {1},      
      //                               {0},     
      //                               {5, 3, 5}},
      // PoolingAVG_BWDTestParamsFloat{{3, 2, 6}, 
      //                               {3, 2, 4},
      //                               {2},
      //                               {2}, 
      //                               {1},  
      //                               {3, 2, 6}}, 
      // PoolingAVG_BWDTestParamsFloat{{4, 2, 6}, 
      //                               {4, 2, 3},
      //                               {2},
      //                               {3}, 
      //                               {1},  
      //                               {4, 2, 6}},
      // // pooling2d
      // PoolingAVG_BWDTestParamsFloat{{5, 3, 5, 4}, // dims of input
      //                               {5, 3, 4, 3},// dims of grad_output
      //                               {2, 2},       // kernel(H, W)
      //                               {1, 1},       // stride(H, W)
      //                               {0, 0},       // padding(H, W)
      //                               {5, 3, 5, 4}},// dims of grad_input
      // PoolingAVG_BWDTestParamsFloat{{5, 3, 5, 4}, 
      //                               {5, 3, 4, 3},
      //                               {2, 2},       
      //                               {},       
      //                               {},      
      //                               {5, 3, 5, 4}},
      // PoolingAVG_BWDTestParamsFloat{{5, 3, 5, 4}, 
      //                               {5, 3, 4, 3},
      //                               {2, 2},       
      //                               {1},      
      //                               {0},     
      //                               {5, 3, 5, 4}},
      // PoolingAVG_BWDTestParamsFloat{{3, 2, 4, 6}, 
      //                               {3, 2, 2, 4},
      //                               {3, 2},
      //                               {2, 2}, 
      //                               {1, 1},  
      //                               {3, 2, 4, 6}}, 
      // PoolingAVG_BWDTestParamsFloat{{4, 2, 6, 6}, 
      //                               {4, 2, 3, 4},
      //                               {2, 3},
      //                               {3, 2}, 
      //                               {1, 2},  
      //                               {4, 2, 6, 6}},
      // // pooling3d
      // PoolingAVG_BWDTestParamsFloat{{5, 3, 5, 4, 5},  
      //                               {5, 3, 4, 3, 4},
      //                               {2, 2, 2},        
      //                               {},             
      //                               {},              
      //                               {5, 3, 5, 4, 5}}, 
      // PoolingAVG_BWDTestParamsFloat{{3, 2, 4, 6, 10}, 
      //                               {3, 2, 2, 4, 5},
      //                               {3, 2, 4},
      //                               {2, 2, 2}, 
      //                               {1, 1, 1},  
      //                               {3, 2, 4, 6, 10}}, 
      // PoolingAVG_BWDTestParamsFloat{{4, 2, 6, 6, 12}, 
      //                               {4, 2, 3, 4, 4},
      //                               {2, 3, 2},
      //                               {3, 2, 4}, 
      //                               {1, 2, 1},  
      //                               {4, 2, 6, 6, 12}}
    )
);



template <typename TScalarType>
struct PoolingMAX_FWDTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> kernel;
  std::vector<int64_t> strides;
  std::vector<int64_t> padding;
  std::vector<int64_t> dims_output;
};

// max fwd outplace
template <typename TScalarType>
class PoolingMAX_FWDTest
    : public ::testing::TestWithParam<PoolingMAX_FWDTestParams<TScalarType>> {};

using PoolingMAX_FWDTestParamsFloat = PoolingMAX_FWDTestParams<float>;
using PoolingMAX_FWDTestFloat = PoolingMAX_FWDTest<float>;

TEST_P(PoolingMAX_FWDTestFloat, Pooling_MAX_FWD_OutplaceTestFloat) {
  PoolingMAX_FWDTestParamsFloat params =
      ::testing::TestWithParam<PoolingMAX_FWDTestParamsFloat>::GetParam();
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  auto cpu_result = pooling_max_fwd(cpu_input, params.kernel, 
                                    params.strides, params.padding);
  Tensor cpu_output = std::get<0>(cpu_result);
  Tensor cpu_indices = std::get<1>(cpu_result);
  // TensorPrinter tp;
  // tp.print(cpu_input);
  // tp.print(cpu_output);
  // tp.print(cpu_indices);

  
  // Tensor cpu_grad_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  // Tensor cpu_grad_input = pooling_max_bwd(cpu_input, cpu_output, cpu_grad_output, cpu_indices,
  //                               params.kernel, params.strides, params.padding);
  // tp.print(cpu_grad_output);
  // tp.print(cpu_grad_input);

  Tensor cuda_input = cpu_input.to(kCUDA);   
  auto cuda_result = pooling_max_fwd(cuda_input, params.kernel, 
                                    params.strides, params.padding);
  Tensor cuda_output = std::get<0>(cuda_result);
  Tensor cuda_indices = std::get<1>(cuda_result);
  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

INSTANTIATE_TEST_CASE_P(
    PoolingMAX_FWDTestFloatSuite, PoolingMAX_FWDTestFloat,
    ::testing::Values(
      // pooling1d
      PoolingMAX_FWDTestParamsFloat{{5, 3, 5}, // dims of input
                                    {2},       // kernel(H, W)
                                    {1},       // stride(H, W)
                                    {0},       // padding(H, W)
                                    {5, 3, 4}},// dims of output
      PoolingMAX_FWDTestParamsFloat{{5, 3, 4}, 
                                    {2},       
                                    {1},      
                                    {0},     
                                    {5, 3, 3}},
      PoolingMAX_FWDTestParamsFloat{{3, 2, 6}, 
                                    {2},
                                    {2}, 
                                    {1},  
                                    {3, 2, 4}}, 
      PoolingMAX_FWDTestParamsFloat{{4, 2, 6}, 
                                    {2},
                                    {3}, 
                                    {1},  
                                    {4, 2, 3}},
      // pooling2d
      PoolingMAX_FWDTestParamsFloat{{5, 3, 5, 4}, // dims of input
                                    {2, 2},       // kernel(H, W)
                                    {1, 1},       // stride(H, W)
                                    {0, 0},       // padding(H, W)
                                    {5, 3, 4, 3}},// dims of output
      PoolingMAX_FWDTestParamsFloat{{5, 3, 5, 4}, 
                                    {2, 2},       
                                    {1},      
                                    {0},     
                                    {5, 3, 4, 3}},
      PoolingMAX_FWDTestParamsFloat{{3, 2, 4, 6}, 
                                    {3, 2},
                                    {2, 2}, 
                                    {1, 1},  
                                    {3, 2, 2, 4}}, 
      PoolingMAX_FWDTestParamsFloat{{4, 2, 6, 6}, 
                                    {2, 3},
                                    {3, 2}, 
                                    {1, 2},  
                                    {4, 2, 3, 4}},
      // pooling3d
      PoolingMAX_FWDTestParamsFloat{{5, 3, 5, 4, 5},  // dims of input
                                    {2, 2, 2},        // kernel(D, H, W)
                                    {},               // stride(D, H, W)
                                    {},               // padding(D, H, W)
                                    {5, 3, 4, 3, 4}}, // dims of output
      PoolingMAX_FWDTestParamsFloat{{3, 2, 4, 6, 10}, 
                                    {3, 2, 4},
                                    {2, 2, 2}, 
                                    {1, 1, 1},  
                                    {3, 2, 2, 4, 5}}, 
      PoolingMAX_FWDTestParamsFloat{{4, 2, 6, 6, 12}, 
                                    {2, 3, 2},
                                    {3, 2, 4}, 
                                    {1, 2, 1},  
                                    {4, 2, 3, 4, 4}}
    )
);






template <typename TScalarType>
struct PoolingMAX_BWDTestParams {
  std::vector<int64_t> dims_input;
  std::vector<int64_t> dims_grad_output;
  std::vector<int64_t> kernel;
  std::vector<int64_t> strides;
  std::vector<int64_t> padding;
  std::vector<int64_t> dims_grad_input;
};

template <typename TScalarType>
class PoolingMAX_BWDTest
    : public ::testing::TestWithParam<PoolingMAX_BWDTestParams<TScalarType>> {};

using PoolingMAX_BWDTestParamsFloat = PoolingMAX_BWDTestParams<float>;
using PoolingMAX_BWDTestFloat = PoolingMAX_BWDTest<float>;

// max bwd outplace
TEST_P(PoolingMAX_BWDTestFloat, Pooling_MAX_BWD_OutplaceTestFloat) {
  PoolingMAX_BWDTestParamsFloat params =
      ::testing::TestWithParam<PoolingMAX_BWDTestParamsFloat>::GetParam();
  
  Tensor cpu_input = rand_uniform(params.dims_input, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  auto cpu_result = pooling_max_fwd(cpu_input, params.kernel, 
                                    params.strides, params.padding);
  Tensor cpu_output = std::get<0>(cpu_result);
  Tensor cpu_indices = std::get<1>(cpu_result);
  Tensor cuda_input = cpu_input.to(kCUDA);   
  auto cuda_result = pooling_max_fwd(cuda_input, params.kernel, 
                                    params.strides, params.padding);
  Tensor cuda_output = std::get<0>(cuda_result);
  Tensor cuda_indices = std::get<1>(cuda_result);

  Tensor cpu_grad_output = rand_uniform(params.dims_grad_output, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_grad_input = pooling_max_bwd(cpu_input, cpu_output,
                                      cpu_grad_output, cpu_indices,
                                      params.kernel, params.strides, 
                                      params.padding);
  // TensorPrinter tp;
  Tensor cuda_grad_output = cpu_grad_output.to(kCUDA);   
  Tensor cuda_grad_input = pooling_max_bwd(cuda_input, cuda_output,
                                      cuda_grad_output, cuda_indices,
                                      params.kernel, params.strides, 
                                      params.padding);
  ExpectEqualDenseRegardlessDevice(cpu_grad_input, cuda_grad_input);
}


INSTANTIATE_TEST_CASE_P(
    PoolingMAX_BWDTestFloatSuite, PoolingMAX_BWDTestFloat,
    ::testing::Values(
      // pooling1d
      PoolingMAX_BWDTestParamsFloat{{5, 3, 5}, // dims of input
                                    {5, 3, 4},// dims of grad_output
                                    {2},       // kernel(H, W)
                                    {1},       // stride(H, W)
                                    {0},       // padding(H, W)
                                    {5, 3, 5}},// dims of grad_input
      PoolingMAX_BWDTestParamsFloat{{5, 3, 4}, 
                                    {5, 3, 3},
                                    {2},       
                                    {},       
                                    {},      
                                    {5, 3, 4}},
      PoolingMAX_BWDTestParamsFloat{{5, 3, 5}, 
                                    {5, 3, 4},
                                    {2},       
                                    {1},      
                                    {0},     
                                    {5, 3, 5}},
      PoolingMAX_BWDTestParamsFloat{{3, 2, 6}, 
                                    {3, 2, 4},
                                    {2},
                                    {2}, 
                                    {1},  
                                    {3, 2, 6}}, 
      PoolingMAX_BWDTestParamsFloat{{4, 2, 6}, 
                                    {4, 2, 3},
                                    {2},
                                    {3}, 
                                    {1},  
                                    {4, 2, 6}},
      // pooling2d
      PoolingMAX_BWDTestParamsFloat{{5, 3, 5, 4}, // dims of input
                                    {5, 3, 4, 3},// dims of grad_output
                                    {2, 2},       // kernel(H, W)
                                    {1, 1},       // stride(H, W)
                                    {0, 0},       // padding(H, W)
                                    {5, 3, 5, 4}},// dims of grad_input
      PoolingMAX_BWDTestParamsFloat{{5, 3, 5, 4}, 
                                    {5, 3, 4, 3},
                                    {2, 2},       
                                    {},       
                                    {},      
                                    {5, 3, 5, 4}},
      PoolingMAX_BWDTestParamsFloat{{5, 3, 5, 4}, 
                                    {5, 3, 4, 3},
                                    {2, 2},       
                                    {1},      
                                    {0},     
                                    {5, 3, 5, 4}},
      PoolingMAX_BWDTestParamsFloat{{3, 2, 4, 6}, 
                                    {3, 2, 2, 4},
                                    {3, 2},
                                    {2, 2}, 
                                    {1, 1},  
                                    {3, 2, 4, 6}}, 
      PoolingMAX_BWDTestParamsFloat{{4, 2, 6, 6}, 
                                    {4, 2, 3, 4},
                                    {2, 3},
                                    {3, 2}, 
                                    {1, 2},  
                                    {4, 2, 6, 6}},
      // pooling3d
      PoolingMAX_BWDTestParamsFloat{{5, 3, 5, 4, 5},  
                                    {5, 3, 4, 3, 4},
                                    {2, 2, 2},        
                                    {},             
                                    {},              
                                    {5, 3, 5, 4, 5}}, 
      PoolingMAX_BWDTestParamsFloat{{3, 2, 4, 6, 10}, 
                                    {3, 2, 2, 4, 5},
                                    {3, 2, 4},
                                    {2, 2, 2}, 
                                    {1, 1, 1},  
                                    {3, 2, 4, 6, 10}}, 
      PoolingMAX_BWDTestParamsFloat{{4, 2, 6, 6, 12}, 
                                    {4, 2, 3, 4, 4},
                                    {2, 3, 2},
                                    {3, 2, 4}, 
                                    {1, 2, 1},  
                                    {4, 2, 6, 6, 12}}
    )
);


// #endif

} // namespace hice
