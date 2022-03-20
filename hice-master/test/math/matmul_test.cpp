#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/basic/transpose.h"
#include "hice/math/matmul.h"

#include "test/tools/compare.h"
#include "gtest/gtest.h"
#include "sys/time.h"
namespace hice {


TEST(MatmulTestFloat, MM) {
  std::vector<int64_t> dims_a = {4, 6};
  std::vector<int64_t> dims_b = {6, 7};
  // cpu
  Tensor tensor_a = rand_normal(dims_a, 1.0, 10.0, device(kCPU).dtype(kFloat));
  Tensor tensor_b = rand_normal(dims_b, 1.0, 10.0, device(kCPU).dtype(kFloat));
        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);

        Tensor output = matmul(tensor_a, tensor_b);
        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;


        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;



//  // gpu
//  Tensor tensor_a_gpu = tensor_a.to(kCUDA);
//  Tensor tensor_b_gpu = tensor_b.to(kCUDA);
//  Tensor output_gpu = matmul(tensor_a_gpu, tensor_b_gpu);

//  ExpectEqualDenseWithError(output, output_gpu, 1e-3);
}

#if 0
template <typename TScalarType>
struct MatmulDenseTestParams {
  std::vector<int64_t> dims_input1;
  std::vector<int64_t> dims_input2;
  std::vector<int64_t> dims_output;
};

template <typename TScalarType>
class MatmulDenseTest
    : public ::testing::TestWithParam<MatmulDenseTestParams<TScalarType>> {};

using MatmulDenseTestParamsFloat = MatmulDenseTestParams<float>;
using MatmulDenseTestFloat = MatmulDenseTest<float>;

TEST_P(MatmulDenseTestFloat, Outplace_Trans11) {
  MatmulDenseTestParamsFloat params =
      ::testing::TestWithParam<MatmulDenseTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));   
  Tensor cpu_output = matmul(transpose_matrix(cpu_input1), transpose_matrix(cpu_input2), 
                            kTrans, kTrans);
  // TensorPrinter tp;
  // tp.print(cpu_input1);
  Tensor cuda_input1 = cpu_input1.to(kCUDA);
  Tensor cuda_input2 = cpu_input2.to(kCUDA);
  Tensor cuda_output = matmul(transpose_matrix(cuda_input1), transpose_matrix(cuda_input2), 
                            kTrans, kTrans);
  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

TEST_P(MatmulDenseTestFloat, Outplace_Trans10) {
  MatmulDenseTestParamsFloat params =
      ::testing::TestWithParam<MatmulDenseTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));   
  Tensor cpu_output = matmul(transpose_matrix(cpu_input1), cpu_input2, 
                            kTrans, kNoTrans);
  // TensorPrinter tp;
  // tp.print(cpu_input1);
  Tensor cuda_input1 = cpu_input1.to(kCUDA);
  Tensor cuda_input2 = cpu_input2.to(kCUDA);
  Tensor cuda_output = matmul(transpose_matrix(cuda_input1), cuda_input2, 
                            kTrans, kNoTrans);
  
  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

TEST_P(MatmulDenseTestFloat, Outplace_Trans01) {
  MatmulDenseTestParamsFloat params =
      ::testing::TestWithParam<MatmulDenseTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));   
  Tensor cpu_output = matmul(cpu_input1, transpose_matrix(cpu_input2), 
                            kNoTrans, kTrans);
  // TensorPrinter tp;
  // tp.print(cpu_input1);
  Tensor cuda_input1 = cpu_input1.to(kCUDA);
  Tensor cuda_input2 = cpu_input2.to(kCUDA);
  Tensor cuda_output = matmul(cuda_input1, transpose_matrix(cuda_input2), 
                            kNoTrans, kTrans);
  
  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

TEST_P(MatmulDenseTestFloat, Outplace_NoTrans) {
  MatmulDenseTestParamsFloat params =
      ::testing::TestWithParam<MatmulDenseTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));   
  Tensor cpu_output = matmul(cpu_input1, cpu_input2, 
                            kNoTrans, kNoTrans);
  // TensorPrinter tp;
  // tp.print(cpu_input1);
  Tensor cuda_input1 = cpu_input1.to(kCUDA);
  Tensor cuda_input2 = cpu_input2.to(kCUDA);
  Tensor cuda_output = matmul(cuda_input1, cuda_input2);
  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

// void natrual_assign(Tensor& tensor) {
//   float* data = tensor.mutable_data<float>();
//   for (size_t i = 0; i < tensor.size(); ++i) {
//     data[i] = i * 0.1;
//   }
// }

TEST_P(MatmulDenseTestFloat, Inplace_NoTrans) {
  MatmulDenseTestParamsFloat params =
      ::testing::TestWithParam<MatmulDenseTestParamsFloat>::GetParam();
  Tensor cpu_input1 = rand_uniform(params.dims_input1, 1.0, 10.0, device(kCPU).dtype(kFloat)); 
  Tensor cpu_input2 = rand_uniform(params.dims_input2, 1.0, 10.0, device(kCPU).dtype(kFloat));   
  Tensor cpu_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCPU).dtype(kFloat));  
  // natrual_assign(cpu_input1);
  // natrual_assign(cpu_input2);
  
  matmul(cpu_input1, cpu_input2, cpu_output);
  // TensorPrinter tp;
  Tensor cuda_input1 = cpu_input1.to(kCUDA);   
  Tensor cuda_input2 = cpu_input2.to(kCUDA);  
  Tensor cuda_output = rand_uniform(params.dims_output, 1.0, 10.0, device(kCUDA).dtype(kFloat));  
  matmul(cuda_input1, cuda_input2, cuda_output);
  ExpectEqualDenseRegardlessDevice(cpu_output, cuda_output);
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    MatmulDenseTestFloatSuite, MatmulDenseTestFloat,
    ::testing::Values(
      // vector-vector
      MatmulDenseTestParamsFloat{{5}, 
                            {5}, 
                            {1}},
      // vector-matrix
      MatmulDenseTestParamsFloat{{5}, 
                            {5, 6}, 
                            {6}},
      // vector-cube
      MatmulDenseTestParamsFloat{{5}, 
                            {4, 5, 6}, 
                            {4, 6}},
      // matrix-vector
      MatmulDenseTestParamsFloat{{5, 6}, 
                            {6}, 
                            {5}},
      // matrix-matrix
      MatmulDenseTestParamsFloat{{5, 6}, 
                            {6, 7}, 
                            {5, 7}},
      // matrix-cube
      MatmulDenseTestParamsFloat{{5, 6}, 
                            {5, 6, 7}, 
                            {5, 5, 7}},
      // cube-vector
      MatmulDenseTestParamsFloat{{5, 6, 7}, 
                            {7}, 
                            {5, 6}},
      // cube-matrix
      MatmulDenseTestParamsFloat{{5, 6, 7}, 
                            {7, 8}, 
                            {5, 6, 8}},
      // hypercube-hypercube(broadcasted batch matmul)
      MatmulDenseTestParamsFloat{{1, 5, 6, 7}, 
                            {4, 1, 7, 2}, 
                            {4, 5, 6, 2}}




      // MatmulDenseTestParamsFloat{{4}, 
      //                       {4}, 
      //                       {1}},
      // MatmulDenseTestParamsFloat{{5, 4}, 
      //                       {4}, 
      //                       {5}},

      // MatmulDenseTestParamsFloat{{5, 4}, 
      //                       {4, 3}, 
      //                       {5, 3}},
      // MatmulDenseTestParamsFloat{{5}, 
      //                       {5, 4}, 
      //                       {4}},
      // MatmulDenseTestParamsFloat{{5, 4, 3}, 
      //                       {3}, 
      //                       {5, 4}},
      // MatmulDenseTestParamsFloat{{3}, 
      //                       {5, 3, 4}, 
      //                       {5, 4}},
      // MatmulDenseTestParamsFloat{{2, 1, 3, 4}, 
      //                       {2, 4, 2}, 
      //                       {2, 2, 3, 2}}
    )
);

#endif

} // namespace hice


