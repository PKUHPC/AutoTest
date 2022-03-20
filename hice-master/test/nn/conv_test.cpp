#include "hice/nn/conv.h"
#include "hice/basic/factories.h"
#include "hice/util/types.h"
#include "hice/core/tensor_printer.h"

#include "test/tools/compare.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "sys/time.h"

namespace hice {
namespace {


TEST(ConvTest, DenseFloat) {
  Tensor input_cpu = rand_uniform({10, 8, 8, 8}, -10, 10, dtype(kFloat).device(kCPU));
  Tensor kernel_cpu = rand_uniform({8, 4, 3, 3}, -10, 10, dtype(kFloat).device(kCPU));
  Tensor bias_cpu = rand_uniform({8}, -10, 10, dtype(kFloat).device(kCPU));
  // Tensor input_cpu = rand_uniform({1, 4, 3, 3}, -2, 2, dtype(kInt32).device(kCPU)).to(kDouble);
  // Tensor kernel_cpu = rand_uniform({4, 2, 2, 2}, -2, 2, dtype(kInt32).device(kCPU)).to(kDouble);
  // Tensor bias_cpu = rand_uniform({4}, -10, 10, dtype(kDouble).device(kCPU));

//  Tensor input_cuda = input_cpu.to(kCUDA);
//  Tensor kernel_cuda = kernel_cpu.to(kCUDA);
//  Tensor bias_cuda = bias_cpu.to(kCUDA);

  std::vector<int64_t> padding = {2, 1};  
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> dilation = {2, 1};
  // std::vector<int64_t> padding = {0, 0};  
  // std::vector<int64_t> stride = {1, 1};
  // std::vector<int64_t> dilation = {1, 1};
  int64_t groups = 2;
//  Tensor output_cuda = conv_fwd(input_cuda, kernel_cuda, bias_cuda, padding, stride, dilation, groups, false, false);
        struct timeval hice_start, hice_end;
        double hice_time;
        gettimeofday(&hice_start,NULL);
        Tensor output_cpu = conv_fwd(input_cpu, kernel_cpu, bias_cpu, padding, stride, dilation, groups, false, false);

        gettimeofday(&hice_end,NULL);
        hice_time = (hice_end.tv_sec - hice_start.tv_sec) * 1000.0
                    + (hice_end.tv_usec - hice_start.tv_usec) / 1000.0 ;
//  Tensor grad_output_cpu = rand_uniform(output_cpu.dims(), -10, 10, dtype(kFloat).device(kCPU));
//  Tensor grad_input_cpu = rand_uniform(input_cpu.dims(), -10, 10, dtype(kFloat).device(kCPU));
//  Tensor grad_kernel_cpu = rand_uniform(kernel_cpu.dims(), -10, 10, dtype(kFloat).device(kCPU));
//  Tensor grad_bias_cpu = rand_uniform(bias_cpu.dims(), -10, 10, dtype(kFloat).device(kCPU));
//  Tensor grad_output_cuda = grad_output_cpu.to(kCUDA);

//  hice::optional<Tensor> grad_input_cuda;
//  hice::optional<Tensor> grad_kernel_cuda;
//  hice::optional<Tensor> grad_bias_cuda;
//  grad_input_cuda = rand_uniform(input_cpu.dims(), -10, 10, dtype(kFloat).device(kCUDA));
//  grad_kernel_cuda = rand_uniform(kernel_cpu.dims(), -10, 10, dtype(kFloat).device(kCUDA));
//  grad_bias_cuda = rand_uniform(bias_cpu.dims(), -10, 10, dtype(kFloat).device(kCUDA));



        std::cout<< /*GREEN <<*/ "\t[ HICE ] " << /*RESET <<*/ hice_time << " ms" << std::endl;
//  conv_bwd(input_cuda, kernel_cuda, grad_output_cuda, padding, stride, dilation, groups, false, false, grad_input_cuda, grad_kernel_cuda, grad_bias_cuda);

  // Tensor grad_input_cpu, grad_kernel_cpu, grad_bias_cpu;
  // std::tie(grad_input_cpu, grad_kernel_cpu, grad_bias_cpu) =
  //     conv_bwd(input_cpu, kernel_cpu, grad_output_cpu, padding, stride, dilation,
  //              groups, false, false, {true, true, true});

  // Tensor grad_input_cuda, grad_kernel_cuda, grad_bias_cuda;
  // std::tie(grad_input_cuda, grad_kernel_cuda, grad_bias_cuda) =
  //     conv_bwd(input_cuda, kernel_cuda, grad_output_cuda, padding, stride, dilation,
  //              groups, false, false, {true, true, true});


//  ExpectEqualDenseRegardlessDevice(output_cpu, output_cuda);
//  ExpectEqualDenseWithError(grad_input_cpu, grad_input_cuda.value());
//  ExpectEqualDenseWithError(grad_kernel_cpu, grad_kernel_cuda.value());
  // ExpectEqualDenseRegardlessDevice(grad_bias_cpu, grad_bias_cuda);
}
  
}
}



#if 0
namespace hice {
namespace {

using ::testing::Pointwise;
using ::testing::FloatEq;
using ::testing::ContainerEq;

struct ConvFwdParams {
  std::vector<int64_t> input_dims;
  std::vector<int64_t> weight_dims;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups;
  ScalarType scalar_type;
};

class ConvFwdTest
    : public ::testing::TestWithParam<ConvFwdParams> {};

TEST_P(ConvFwdTest, CompareCPUWithCUDA) {
  ConvFwdParams params = ::testing::TestWithParam<ConvFwdParams>::GetParam();
  std::vector<int64_t> input_dims = params.input_dims;
  std::vector<int64_t> weight_dims = params.weight_dims;
  std::vector<int64_t> bias_dims{weight_dims.at(0)};
  std::vector<int64_t> padding = params.padding;
  std::vector<int64_t> stride = params.stride;
  std::vector<int64_t> dilation = params.dilation;
  int64_t groups = params.groups;
  ScalarType scalar_type = params.scalar_type;
  Tensor h_input =
      rand_uniform(input_dims, 1.0, 4.0, device(kCPU).dtype(scalar_type));
  Tensor h_weight =
      rand_uniform(weight_dims, 1.0, 4.0, device(kCPU).dtype(scalar_type));
  Tensor h_bias =
      rand_uniform(bias_dims, 1.0, 4.0, device(kCPU).dtype(scalar_type));
  Tensor h_output = conv_fwd(h_input, h_weight, h_bias, padding, stride,
                             dilation, groups, true, true);
  
  Tensor d_input = h_input.to(kCUDA);
  Tensor d_weight = h_weight.to(kCUDA);
  Tensor d_bias = h_bias.to(kCUDA);
  Tensor d_output = conv_fwd(d_input, d_weight, d_bias, padding, stride,
                             dilation, groups, true, true).to(kCPU);
  //TensorPrinter tp;
  //tp.print(h_output);
  //tp.print(d_output);
  HICE_DISPATCH_ALL_TYPES(scalar_type, "conv fwd test", [&] {
    //using scalar_t = float;
    hice::ArrayRef<scalar_t> h_output_data(h_output.mutable_data<scalar_t>(),
                                       h_output.size());
    hice::ArrayRef<scalar_t> d_output_data(d_output.mutable_data<scalar_t>(),
                                       d_output.size());
    if (std::is_floating_point<scalar_t>::value) {
      for (int i = 0; i < h_output.size(); ++i) {
        scalar_t diff = h_output_data[i] - d_output_data[i];
        scalar_t e = (std::abs(h_output_data[i]) > 1e-4)
                         ? (diff / h_output_data[i])
                         : diff;
        EXPECT_NEAR(e, (scalar_t)0.0, 1e-4)
            << "Index: " << i << " Value: " << h_output_data[i];
      }
      //EXPECT_THAT(h_output_data, Pointwise(FloatEq(), d_output_data));
    } else if (std::is_integral<scalar_t>::value) {
      EXPECT_THAT(h_output_data, ContainerEq(d_output_data));
    } else {
    }
  });
}

//"INSTANTIATE_TEST_CASE_P" will be deprecated and use
//"INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    ConvFwdTestSuite, ConvFwdTest,
    ::testing::Values(
      ConvFwdParams{
        {1, 1, 5, 5}, {1, 1, 3, 3}, {0, 0}, {1, 1}, {2, 2}, 1, kFloat},
      ConvFwdParams{
        {1, 2, 8, 10}, {1, 2, 4, 5}, {1, 1}, {3, 3}, {2, 2}, 1, kFloat},
      ConvFwdParams{
        {3, 3, 33, 55}, {1, 3, 5, 7}, {2, 2}, {3, 3}, {3, 3}, 1, kFloat},
      ConvFwdParams{
        {3, 8, 16, 20}, {4, 4, 3, 4}, {1, 1}, {3, 3}, {1, 1}, 2, kFloat},
      ConvFwdParams{
        {2, 4, 30, 30}, {1, 4, 5, 5}, {0, 0}, {4, 4}, {1, 1}, 1, kFloat},
      ConvFwdParams{
        {2, 3, 227, 227}, {96, 3, 11, 11}, {0, 0}, {4, 4}, {1, 1}, 1, kFloat}
  ));

struct ConvBwdParams {
  std::vector<int64_t> input_dims;
  std::vector<int64_t> weight_dims;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups;
  ScalarType scalar_type;
};

class ConvBwdTest
    : public ::testing::TestWithParam<ConvBwdParams> {};

TEST_P(ConvBwdTest, CompareCPUWithCUDA) {
  ConvBwdParams params = ::testing::TestWithParam<ConvBwdParams>::GetParam();
  std::vector<int64_t> input_dims = params.input_dims;
  std::vector<int64_t> weight_dims = params.weight_dims;
  std::vector<int64_t> bias_dims{weight_dims.at(0)};
  std::vector<int64_t> padding = params.padding;
  std::vector<int64_t> stride = params.stride;
  std::vector<int64_t> dilation = params.dilation;
  int64_t groups = params.groups;
  ScalarType scalar_type = params.scalar_type;
  auto output_dims = hice::conv_output_dims(input_dims, weight_dims, padding,
                                            stride, dilation, groups);
  Tensor h_input =
      rand_uniform(input_dims, 1.0, 4.0, device(kCPU).dtype(scalar_type));
  Tensor h_weight =
      rand_uniform(weight_dims, 1.0, 4.0, device(kCPU).dtype(scalar_type));
  Tensor h_grad_output =
      rand_uniform(output_dims, 1.0, 4.0, device(kCPU).dtype(scalar_type));
  Tensor h_grad_input, h_grad_weight, h_grad_bias;
  std::tie(h_grad_input, h_grad_weight, h_grad_bias) =
      conv_bwd(h_input, h_weight, h_grad_output, padding, stride, dilation,
               groups, true, true, {true, true, true});

  Tensor d_input = h_input.to(kCUDA);
  Tensor d_weight = h_weight.to(kCUDA);
  Tensor d_grad_output = h_grad_output.to(kCUDA);
  Tensor d_grad_input0, d_grad_weight0, d_grad_bias0;
  std::tie(d_grad_input0, d_grad_weight0, d_grad_bias0) =
      conv_bwd(d_input, d_weight, d_grad_output, padding, stride, dilation,
               groups, true, true, {true, true, true});
  Tensor d_grad_input = d_grad_input0.to(kCPU);
  Tensor d_grad_weight = d_grad_weight0.to(kCPU);
  Tensor d_grad_bias = d_grad_bias0.to(kCPU);
  TensorPrinter tp;
  //tp.print(h_grad_input);
  //tp.print(d_grad_input);
  //tp.print(h_grad_weight);
  //tp.print(d_grad_weight);
  //tp.print(h_grad_bias);
  //tp.print(d_grad_bias);
  HICE_DISPATCH_ALL_TYPES(scalar_type, "conv bwd test", [&] {
    //using scalar_t = float;
    hice::ArrayRef<scalar_t> h_grad_input_data(h_grad_input.mutable_data<scalar_t>(),
                                           h_grad_input.size());
    hice::ArrayRef<scalar_t> d_grad_input_data(d_grad_input.mutable_data<scalar_t>(),
                                           d_grad_input.size());
    hice::ArrayRef<scalar_t> h_grad_weight_data(h_grad_weight.mutable_data<scalar_t>(),
                                            h_grad_weight.size());
    hice::ArrayRef<scalar_t> d_grad_weight_data(d_grad_weight.mutable_data<scalar_t>(),
                                            d_grad_input.size());
    hice::ArrayRef<scalar_t> h_grad_bias_data(h_grad_bias.mutable_data<scalar_t>(),
                                          h_grad_bias.size());
    hice::ArrayRef<scalar_t> d_grad_bias_data(d_grad_bias.mutable_data<scalar_t>(),
                                          d_grad_input.size());
    if (std::is_floating_point<scalar_t>::value) {
      for (int i = 0; i < h_grad_input.size(); ++i) {
        scalar_t diff = h_grad_input_data[i] - d_grad_input_data[i];
        //if (std::isnan(diff) || std::isinf(diff)) continue;
        scalar_t e = (std::abs(h_grad_input_data[i]) > 1e-4)
                         ? (diff / h_grad_input_data[i])
                         : diff;
        EXPECT_NEAR(e, (scalar_t)0.0, 1e-4)
            << "Grad_input -> index: " << i
            << " Value 1: " << h_grad_input_data[i]
            << " value 2: " << d_grad_input_data[i];
      }

      for (int i = 0; i < h_grad_weight.size(); ++i) {
        //if (std::isnan(h_grad_weight_data[i]) ||
        //    std::isinf(h_grad_weight_data[i]) ||
        //    std::isnan(d_grad_weight_data[i]) ||
        //    std::isinf(d_grad_weight_data[i])) {
        //  std::cout << "NaN or INF->Index : " << i
        //            << " Value 1: " << h_grad_weight_data[i]
        //            << " value 2: " << d_grad_weight_data[i] << std::endl;
        //  continue;
        //}
        scalar_t diff = h_grad_weight_data[i] - d_grad_weight_data[i];
        //if (std::isnan(diff) || std::isinf(diff)) continue;
        scalar_t e = (std::abs(h_grad_weight_data[i]) > 1e-4)
                         ? (diff / h_grad_weight_data[i])
                         : diff;
        EXPECT_NEAR(e, (scalar_t)0.0, 1e-4)
            << "Grad_weight -> Index: " << i
            << " Value 1: " << h_grad_weight_data[i]
            << " value 2: " << d_grad_weight_data[i];
      }

      for (int i = 0; i < h_grad_bias.size(); ++i) {
        scalar_t diff = h_grad_bias_data[i] - d_grad_bias_data[i];
        //if (std::isnan(diff) || std::isinf(diff)) continue;
        scalar_t e = (std::abs(h_grad_bias_data[i]) > 1e-4)
                         ? (diff / h_grad_bias_data[i])
                         : diff;
        EXPECT_NEAR(e, (scalar_t)0.0, 1e-4)
            << "Grad_bias -> Index: " << i << " Value: " << h_grad_bias_data[i];
      }
      //EXPECT_THAT(h_output_data, Pointwise(FloatEq(), d_output_data));
    } else if (std::is_integral<scalar_t>::value) {
    } else {
    }
  });
}

// "INSTANTIATE_TEST_CASE_P" will be deprecated and use
// "INSTANTIATE_TEST_SUITE_P" instead in the future
INSTANTIATE_TEST_CASE_P(
    ConvBwdTestSuite, ConvBwdTest,
    ::testing::Values(
      ConvBwdParams{
        {1, 1, 5, 5}, {1, 1, 3, 3}, {0, 0}, {1, 1}, {2, 2}, 1, kFloat},
      ConvBwdParams{
        {1, 2, 8, 10}, {1, 2, 4, 5}, {1, 1}, {3, 3}, {2, 2}, 1, kFloat},
      ConvBwdParams{
        {3, 3, 33, 55}, {1, 3, 5, 7}, {2, 2}, {3, 3}, {2, 2}, 1, kFloat},
      ConvBwdParams{
        {3, 8, 16, 20}, {4, 4, 3, 4}, {1, 1}, {3, 3}, {1, 1}, 2, kFloat},
      ConvBwdParams{
        {2, 4, 30, 30}, {1, 4, 5, 5}, {0, 0}, {4, 4}, {1, 1}, 1, kFloat},
      ConvBwdParams{
        {2, 3, 227, 227}, {96, 3, 11, 11}, {0, 0}, {4, 4}, {1, 1}, 1, kFloat}
  ));
}
}  // namespace hice

#endif 