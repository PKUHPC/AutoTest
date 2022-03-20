#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/ml/svm.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"

namespace hice {

template <typename TScalarType>
struct SvmTestParams {
  std::vector<int64_t> dims_train_data;
  std::vector<int64_t> dims_label;
  std::vector<int64_t> dims_predict_data;
  std::vector<int64_t> dims_result;
};

template <typename TScalarType>
class SvmTest
    : public ::testing::TestWithParam<SvmTestParams<TScalarType>> {};

using SvmTestParamsFloat = SvmTestParams<float>;
using SvmTestFloat = SvmTest<float>;
TEST_P(SvmTestFloat, svm) {
  SvmTestParamsFloat params =
      ::testing::TestWithParam<SvmTestParamsFloat>::GetParam();
  // TensorPrinter tp;
//cpu
  Tensor cpu_train_data = rand_uniform(params.dims_train_data, -1.0, +1.0, device(kCPU).dtype(kFloat));
  Tensor cpu_label = full(params.dims_label, 0, device(kCPU).dtype(kInt32));
  Tensor cpu_predict_data = wrap(params.dims_train_data, cpu_train_data.mutable_data<float>(), device(kCPU).dtype(kFloat), true);
  Tensor cpu_result = full(params.dims_result, 0, device(kCPU).dtype(kInt32));
  Tensor cpu_result_outplace;

  int *label_data = cpu_label.mutable_data<int>();
  for (int i = 0; i < cpu_label.size() / 2; i++)
    label_data[i] = 1;
  for (int i = cpu_label.size() / 2; i < cpu_label.size(); i++)
    label_data[i] = -1;
  SvmParam param;
  //inplace
  svm(cpu_train_data, cpu_label, cpu_predict_data, cpu_result, param);
  //outplace
  cpu_result_outplace = svm(cpu_train_data, cpu_label, cpu_predict_data, param);

//cuda
  Tensor cuda_train_data = cpu_train_data.to(kCUDA);
  Tensor cuda_label = cpu_label.to(kCUDA);
  Tensor cuda_predict_data = cpu_predict_data.to(kCUDA);
  Tensor cuda_result =  full(params.dims_result, 0, device(kCUDA).dtype(kInt32));
  Tensor cuda_result_outplace;
  //inplace
  svm(cuda_train_data, cuda_label, cuda_predict_data, cuda_result, param);
  //outplace
  cuda_result_outplace = svm(cuda_train_data, cuda_label, cuda_predict_data, param);
  // data compare(cpu, cuda)
  ExpectEqualDenseRegardlessDevice(cpu_result, cuda_result);
  ExpectEqualDenseRegardlessDevice(cpu_result, cpu_result_outplace);
  ExpectEqualDenseRegardlessDevice(cpu_result, cuda_result_outplace);
}

INSTANTIATE_TEST_CASE_P(
    SvmTestFloatSuite, SvmTestFloat,
    ::testing::Values(
      SvmTestParamsFloat{{100, 30},
                         {100},
                         {100, 30},
                         {100}},
      SvmTestParamsFloat{{200, 10},
                         {200},
                         {200, 10},
                         {200}}
      // SvmTestParamsFloat{{3000, 50},
      //                    {3000},
      //                    {3000, 50},
      //                    {3000}},
      // SvmTestParamsFloat{{10000, 100},
      //                    {10000},
      //                    {10000, 100},
      //                    {10000}}
    )
);

// #endif
} // namespace hice
