#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/math/compare.h"

#include "gtest/gtest.h"

namespace hice {

// binary Tensor-Tensor
using CompareTTInplaceOp = std::function<Tensor(Tensor& a, Tensor& b, Tensor& c)>;
auto EQUALTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return equal(a, b, c); };
auto LESS_EQUALTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return less_equal(a, b, c); };
auto LESSTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return less(a, b, c); };
auto GREATERTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return greater(a, b, c); };
auto GREATER_EQUALTT_INPLACE = [](Tensor& a, Tensor& b, Tensor& c)->Tensor{ return greater_equal(a, b, c); };

template <typename TScalarType>
struct CompareTTInplaceTestParams {
  std::vector<int64_t> dims_input1;
  std::vector<int64_t> dims_input2;
  std::vector<int64_t> dims_output;
  CompareTTInplaceOp Op;
};

template <typename TScalarType>
class CompareTTInplaceTest
    : public ::testing::TestWithParam<CompareTTInplaceTestParams<TScalarType>> {};

using CompareTTInplaceTestParamsFloat = CompareTTInplaceTestParams<float>;
using CompareTTInplaceTestFloat = CompareTTInplaceTest<float>;
TEST_P(CompareTTInplaceTestFloat, RandUniformValueTests) {
  CompareTTInplaceTestParamsFloat params =
      ::testing::TestWithParam<CompareTTInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));   
  Tensor cpu_output = empty(params.dims_output, device(kCPU).dtype(kBool));
  Tensor cuda_output = empty(params.dims_output, device(kCUDA).dtype(kBool)); 
  params.Op(cpu_input1, cpu_input2, cpu_output);
  Tensor cuda_input1 = cpu_input1.to(kCUDA);   
  Tensor cuda_input2 = cpu_input2.to(kCUDA);  
  params.Op(cuda_input1, cuda_input2, cuda_output);
  // dims compare(cpu, cuda, Truth)
  auto ndim_output_truth = params.dims_output.size();
  EXPECT_EQ(cpu_output.ndim(), ndim_output_truth);
  EXPECT_EQ(cuda_output.ndim(), ndim_output_truth);
  for (size_t i = 0; i < ndim_output_truth; ++i) {
    EXPECT_EQ(cpu_output.dim(i), params.dims_output[i]);
    EXPECT_EQ(cuda_output.dim(i), params.dims_output[i]);
  }
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_output.size();
  auto size_cuda_output = cuda_output.size();
  Tensor cuda_output_host = cuda_output.to(kCPU);
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  for(int i = 0; i < size_cpu_output; ++i) {
    EXPECT_EQ(cpu_output.data<bool>()[i], cuda_output_host.data<bool>()[i]);
  }
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    CompareTTInplaceTestFloatSuite, CompareTTInplaceTestFloat,
    ::testing::Values(
      // none-broadcast
      CompareTTInplaceTestParamsFloat{{}, 
                                    {}, 
                                    {},
                                    EQUALTT_INPLACE},
      CompareTTInplaceTestParamsFloat{{5}, 
                                    {5}, 
                                    {5},
                                    EQUALTT_INPLACE},
      CompareTTInplaceTestParamsFloat{{6}, 
                                    {6}, 
                                    {6},
                                    LESS_EQUALTT_INPLACE},
      CompareTTInplaceTestParamsFloat{{8}, 
                                    {8}, 
                                    {8},
                                    LESSTT_INPLACE},
      CompareTTInplaceTestParamsFloat{{1}, 
                                    {1}, 
                                    {1},
                                    GREATERTT_INPLACE},
      CompareTTInplaceTestParamsFloat{{3}, 
                                    {3}, 
                                    {3},
                                    GREATER_EQUALTT_INPLACE},
      // broadcast
      CompareTTInplaceTestParamsFloat{{}, 
                                    {5, 6, 7}, 
                                    {5, 6, 7},
                                    EQUALTT_INPLACE},
      CompareTTInplaceTestParamsFloat{{5, 6, 7}, 
                                    {}, 
                                    {5, 6, 7},
                                    EQUALTT_INPLACE},
      CompareTTInplaceTestParamsFloat{{1, 6, 7}, 
                                    {5, 6, 7}, 
                                    {5, 6, 7},
                                    EQUALTT_INPLACE},
      CompareTTInplaceTestParamsFloat{{1, 1, 1}, 
                                    {2, 3, 5}, 
                                    {2, 3, 5},
                                    LESS_EQUALTT_INPLACE},
      CompareTTInplaceTestParamsFloat{{1, 6, 1, 4}, 
                                    {2, 1, 3, 1}, 
                                    {2, 6, 3, 4},
                                    LESSTT_INPLACE},
      CompareTTInplaceTestParamsFloat{         {2}, 
                                    {2, 3, 1, 2}, 
                                    {2, 3, 1, 2},
                                    GREATERTT_INPLACE},
      // broadcast && parallel
      CompareTTInplaceTestParamsFloat{{1, 60, 70}, 
                                    {8, 60, 70}, 
                                    {8, 60, 70},
                                    GREATER_EQUALTT_INPLACE}
      )
);




// binary Tensor-Scalar(numerical)
using CompareTSInplaceOp = std::function<Tensor(Tensor& a, Scalar b, Tensor& c)>;
auto EQUALTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return equal(a, b, c); };
auto LESS_EQUALTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return less_equal(a, b, c); };
auto LESSTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return less(a, b, c); };
auto GREATERTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return greater(a, b, c); };
auto GREATER_EQUALTS_INPLACE = [](Tensor& a, Scalar b, Tensor& c)->Tensor{ return greater_equal(a, b, c); };

template <typename TScalarType>
struct CompareTSInplaceTestParams {
  std::vector<int64_t> dims_input1;
  std::vector<int64_t> dims_output;
  CompareTSInplaceOp Op;
};

template <typename TScalarType>
class CompareTSInplaceTest
    : public ::testing::TestWithParam<CompareTSInplaceTestParams<TScalarType>> {};

using CompareTSInplaceTestParamsFloat = CompareTSInplaceTestParams<float>;
using CompareTSInplaceTestFloat = CompareTSInplaceTest<float>;
TEST_P(CompareTSInplaceTestFloat, RandUniformValueTests) {
  CompareTSInplaceTestParamsFloat params =
      ::testing::TestWithParam<CompareTSInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cpu_output = full(params.dims_output, false, device(kCPU).dtype(kBool));  
  Tensor cuda_input1 = cpu_input1.to(kCUDA);
  Tensor cuda_output = full(params.dims_output, false, device(kCUDA).dtype(kBool)); 
  float scalar = cpu_input1.data<float>()[0]; 
  params.Op(cpu_input1, scalar, cpu_output);
  params.Op(cuda_input1, scalar, cuda_output);
  // dims compare(cpu, cuda, Truth)
  auto ndim_output_truth = params.dims_output.size();
  EXPECT_EQ(cpu_output.ndim(), ndim_output_truth);
  EXPECT_EQ(cuda_output.ndim(), ndim_output_truth);
  for (size_t i = 0; i < ndim_output_truth; ++i) {
    EXPECT_EQ(cpu_output.dim(i), params.dims_output[i]);
    EXPECT_EQ(cuda_output.dim(i), params.dims_output[i]);
  }
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_output.size();
  auto size_cuda_output = cuda_output.size();
  Tensor cuda_output_host = cuda_output.to(kCPU);
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  for(int i = 0; i < size_cpu_output; ++i) {
    EXPECT_EQ(cpu_output.data<bool>()[i], cuda_output_host.data<bool>()[i]);
  }
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    CompareTSInplaceTestFloatSuite, CompareTSInplaceTestFloat,
    ::testing::Values(
      // broadcast
      CompareTSInplaceTestParamsFloat{{5}, 
                                    {5},
                                    EQUALTS_INPLACE},
      CompareTSInplaceTestParamsFloat{{2, 3, 1, 2}, 
                                    {2, 3, 1, 2},
                                    LESS_EQUALTS_INPLACE},
      CompareTSInplaceTestParamsFloat{{2, 6, 3, 4}, 
                                    {2, 6, 3, 4},
                                    LESSTS_INPLACE},
      CompareTSInplaceTestParamsFloat{{2, 3, 5}, 
                                    {2, 3, 5},
                                    GREATERTS_INPLACE},
      CompareTSInplaceTestParamsFloat{{5, 6, 7}, 
                                    {5, 6, 7},
                                    GREATER_EQUALTS_INPLACE},
      // broadcast && parallel
      CompareTSInplaceTestParamsFloat{{8, 60, 70}, 
                                    {8, 60, 70},
                                    GREATER_EQUALTS_INPLACE}
    )
);




// bianry Scalar-Tensor
using CompareSTInplaceOp = std::function<Tensor(Scalar a, Tensor& b, Tensor& c)>;
auto EQUALST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return equal(a, b, c); };
auto LESS_EQUALST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return less_equal(a, b, c); };
auto LESSST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return less(a, b, c); };
auto GREATERST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return greater(a, b, c); };
auto GREATER_EQUALST_INPLACE = [](Scalar a, Tensor& b, Tensor& c)->Tensor{ return greater_equal(a, b, c); };

template <typename TScalarType>
struct CompareSTInplaceTestParams {
  std::vector<int64_t> dims_input2;
  std::vector<int64_t> dims_output;
  CompareSTInplaceOp Op;
};

template <typename TScalarType>
class CompareSTInplaceTest
    : public ::testing::TestWithParam<CompareSTInplaceTestParams<TScalarType>> {};

using CompareSTInplaceTestParamsFloat = CompareSTInplaceTestParams<float>;
using CompareSTInplaceTestFloat = CompareSTInplaceTest<float>;
TEST_P(CompareSTInplaceTestFloat, RandUniformValueTests) {
  CompareSTInplaceTestParamsFloat params =
      ::testing::TestWithParam<CompareSTInplaceTestParamsFloat>::GetParam();
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor cuda_input2 = cpu_input2.to(kCUDA);
  Tensor cpu_output = full(params.dims_output, false, device(kCPU).dtype(kBool)); 
  Tensor cuda_output = full(params.dims_output, false, device(kCUDA).dtype(kBool)); 
  float scalar = cpu_input2.data<float>()[0]; 
  params.Op(scalar, cpu_input2, cpu_output);
  params.Op(scalar, cuda_input2, cuda_output);
  // dims compare(cpu, cuda, Truth)
  auto ndim_output_truth = params.dims_output.size();
  EXPECT_EQ(cpu_output.ndim(), ndim_output_truth);
  EXPECT_EQ(cuda_output.ndim(), ndim_output_truth);
  for (size_t i = 0; i < ndim_output_truth; ++i) {
    EXPECT_EQ(cpu_output.dim(i), params.dims_output[i]);
    EXPECT_EQ(cuda_output.dim(i), params.dims_output[i]);
  }
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_output.size();
  auto size_cuda_output = cuda_output.size();
  Tensor cuda_output_host = cuda_output.to(kCPU);
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  for(int i = 0; i < size_cpu_output; ++i) {
    EXPECT_EQ(cpu_output.data<bool>()[i], cuda_output_host.data<bool>()[i]);
  }
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    CompareSTInplaceTestFloatSuite, CompareSTInplaceTestFloat,
    ::testing::Values(
      // broadcast
      CompareSTInplaceTestParamsFloat{{5}, 
                                    {5},
                                    EQUALST_INPLACE},
      CompareSTInplaceTestParamsFloat{{2, 3, 1, 2}, 
                                    {2, 3, 1, 2},
                                    LESS_EQUALST_INPLACE},
      CompareSTInplaceTestParamsFloat{{2, 6, 3, 4}, 
                                    {2, 6, 3, 4},
                                    LESSST_INPLACE},
      CompareSTInplaceTestParamsFloat{{2, 3, 5}, 
                                    {2, 3, 5},
                                    GREATERST_INPLACE},
      CompareSTInplaceTestParamsFloat{{5, 6, 7}, 
                                    {5, 6, 7},
                                    GREATER_EQUALST_INPLACE},
      // broadcast && parallel
      CompareSTInplaceTestParamsFloat{{8, 60, 70}, 
                                    {8, 60, 70},
                                    GREATER_EQUALST_INPLACE}
    )
);

// #endif
} // namespace hice
 
