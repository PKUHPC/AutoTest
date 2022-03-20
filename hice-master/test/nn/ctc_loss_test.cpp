#include "hice/nn/ctc_loss.h"
#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"

#include <cstring>

#include "gtest/gtest.h"


namespace hice {

template <typename TScalarType>
struct CTCLossInplaceTestParams {
  int64_t batch_size;
  int64_t max_time;
  int64_t max_length;
  int64_t n_classes;
};

using scalar_t = double;
#define kDataType kDouble

template <typename TScalarType>
class CTCLossInplaceTest
    : public ::testing::TestWithParam<CTCLossInplaceTestParams<TScalarType>> {};

using CTCLossInplaceTestParamsDouble = CTCLossInplaceTestParams<scalar_t>;
using CTCLossInplaceTestDouble = CTCLossInplaceTest<scalar_t>;
// #if 0
TEST_P(CTCLossInplaceTestDouble, None) {
  CTCLossInplaceTestParamsDouble params =
      ::testing::TestWithParam<CTCLossInplaceTestParamsDouble>::GetParam();
  int64_t batch_size = params.batch_size;
  int64_t max_time = params.max_time;
  int64_t max_length = params.max_length;
  int64_t n_classes = params.n_classes;
  // cpu inputs
  Tensor cpu_probs = rand_uniform({max_time, batch_size, n_classes}, 0.0, 1.0,
                                  device(kCPU).dtype(kDataType));
  Tensor cpu_target = rand_uniform({batch_size, max_length}, 1, n_classes,
                                   device(kCPU).dtype(kInt32));
  Tensor cpu_probs_lengths =
      full({batch_size}, max_time, device(kCPU).dtype(kInt32));
  Tensor cpu_target_lengths =
      rand_uniform({batch_size}, 0, max_length, device(kCPU).dtype(kInt32));
  Tensor cpu_grad_loss =
      rand_uniform({batch_size}, 0.0, 10.0, device(kCPU).dtype(kDataType));
  // gpu inputs
  Tensor cuda_probs = cpu_probs.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_probs_lengths = cpu_probs_lengths.to(kCUDA);
  Tensor cuda_target_lengths = cpu_target_lengths.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  // fwd
  std::tuple<Tensor, Tensor> cpu_res =
      ctc_loss_fwd(cpu_probs, cpu_target, cpu_probs_lengths, cpu_target_lengths,
                   Reduction::none);
  std::tuple<Tensor, Tensor> cuda_res =
      ctc_loss_fwd(cuda_probs, cuda_target, cuda_probs_lengths,
                   cuda_target_lengths, Reduction::none);
  Tensor cpu_loss = std::get<0>(cpu_res);
  Tensor cuda_loss = std::get<0>(cuda_res);
  Tensor cpu_log_alphas = std::get<1>(cpu_res);
  Tensor cuda_log_alphas = std::get<1>(cuda_res);
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_loss.size();
  auto size_cuda_output = cuda_loss.size();
  Tensor cuda_output_host = cuda_loss.to(kCPU);
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  for (int i = 0; i < size_cpu_output; ++i) {
    EXPECT_LE(std::abs(cpu_loss.data<scalar_t>()[i] - cuda_output_host.data<scalar_t>()[i]), 1e-1);
  }
  // bwd
  Tensor cpu_grad_probs =
      ctc_loss_bwd(cpu_probs, cpu_target, cpu_probs_lengths, cpu_target_lengths,
                   Reduction::none, cpu_log_alphas, cpu_grad_loss);
  Tensor cuda_grad_probs = ctc_loss_bwd(
      cuda_probs, cuda_target, cuda_probs_lengths, cuda_target_lengths,
      Reduction::none, cuda_log_alphas, cuda_grad_loss);
  // data compare(cpu, cuda)
  size_cpu_output = cpu_grad_probs.size();
  size_cuda_output = cuda_grad_probs.size();
  cuda_output_host = cuda_grad_probs.to(kCPU);
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  for (int i = 0; i < size_cpu_output; ++i) {
    EXPECT_LE(
        std::abs(cpu_grad_probs.data<scalar_t>()[i] - cuda_output_host.data<scalar_t>()[i]),
        1e-1);
  }
  // TensorPrinter tp;
  // tp.print(cuda_target);
  // tp.print(cpu_probs_lengths);
  // tp.print(cpu_target_lengths);
  // tp.print(cpu_grad_probs);
  // tp.print(cuda_grad_probs);
}
// #endif

// #if 0
TEST_P(CTCLossInplaceTestDouble, Sum) {
  CTCLossInplaceTestParamsDouble params =
      ::testing::TestWithParam<CTCLossInplaceTestParamsDouble>::GetParam();
  int64_t batch_size = params.batch_size;
  int64_t max_time = params.max_time;
  int64_t max_length = params.max_length;
  int64_t n_classes = params.n_classes;
  // cpu inputs
  Tensor cpu_probs = rand_uniform({max_time, batch_size, n_classes}, 0.0, 1.0,
                                  device(kCPU).dtype(kDataType));
  Tensor cpu_target = rand_uniform({batch_size, max_length}, 1, n_classes,
                                   device(kCPU).dtype(kInt32));
  Tensor cpu_probs_lengths =
      full({batch_size}, max_time, device(kCPU).dtype(kInt32));
  Tensor cpu_target_lengths =
      rand_uniform({batch_size}, 0, max_length, device(kCPU).dtype(kInt32));
  Tensor cpu_grad_loss =
      rand_uniform({}, 0.0, 10.0, device(kCPU).dtype(kDataType));
  // gpu inputs
  Tensor cuda_probs = cpu_probs.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_probs_lengths = cpu_probs_lengths.to(kCUDA);
  Tensor cuda_target_lengths = cpu_target_lengths.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  // fwd
  std::tuple<Tensor, Tensor> cpu_res =
      ctc_loss_fwd(cpu_probs, cpu_target, cpu_probs_lengths, cpu_target_lengths,
                   Reduction::sum);
  std::tuple<Tensor, Tensor> cuda_res =
      ctc_loss_fwd(cuda_probs, cuda_target, cuda_probs_lengths,
                   cuda_target_lengths, Reduction::sum);
  Tensor cpu_loss = std::get<0>(cpu_res);
  Tensor cuda_loss = std::get<0>(cuda_res);
  Tensor cpu_log_alphas = std::get<1>(cpu_res);
  Tensor cuda_log_alphas = std::get<1>(cuda_res);
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_loss.size();
  auto size_cuda_output = cuda_loss.size();
  Tensor cuda_output_host = cuda_loss.to(kCPU);
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  for (int i = 0; i < size_cpu_output; ++i) {
    EXPECT_LE(std::abs(cpu_loss.data<scalar_t>()[i] - cuda_output_host.data<scalar_t>()[i]), 1e-1);
  }
  // bwd
  Tensor cpu_grad_probs =
      ctc_loss_bwd(cpu_probs, cpu_target, cpu_probs_lengths, cpu_target_lengths,
                   Reduction::sum, cpu_log_alphas, cpu_grad_loss);
  Tensor cuda_grad_probs = ctc_loss_bwd(
      cuda_probs, cuda_target, cuda_probs_lengths, cuda_target_lengths,
      Reduction::sum, cuda_log_alphas, cuda_grad_loss);
  // data compare(cpu, cuda)
  size_cpu_output = cpu_grad_probs.size();
  size_cuda_output = cuda_grad_probs.size();
  cuda_output_host = cuda_grad_probs.to(kCPU);
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  for (int i = 0; i < size_cpu_output; ++i) {
    EXPECT_LE(
        std::abs(cpu_grad_probs.data<scalar_t>()[i] - cuda_output_host.data<scalar_t>()[i]),
        1e-1);
  }
}

TEST_P(CTCLossInplaceTestDouble, Mean) {
  CTCLossInplaceTestParamsDouble params =
      ::testing::TestWithParam<CTCLossInplaceTestParamsDouble>::GetParam();
  int64_t batch_size = params.batch_size;
  int64_t max_time = params.max_time;
  int64_t max_length = params.max_length;
  int64_t n_classes = params.n_classes;
  // cpu inputs
  Tensor cpu_probs = rand_uniform({max_time, batch_size, n_classes}, 0.0, 1.0,
                                  device(kCPU).dtype(kDataType));
  Tensor cpu_target = rand_uniform({batch_size, max_length}, 1, n_classes,
                                   device(kCPU).dtype(kInt32));
  Tensor cpu_probs_lengths =
      full({batch_size}, max_time, device(kCPU).dtype(kInt32));
  Tensor cpu_target_lengths =
      rand_uniform({batch_size}, 0, max_length, device(kCPU).dtype(kInt32));
  Tensor cpu_grad_loss =
      rand_uniform({}, 0.0, 10.0, device(kCPU).dtype(kDataType));
  // gpu inputs
  Tensor cuda_probs = cpu_probs.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_probs_lengths = cpu_probs_lengths.to(kCUDA);
  Tensor cuda_target_lengths = cpu_target_lengths.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);
  // fwd
  std::tuple<Tensor, Tensor> cpu_res =
      ctc_loss_fwd(cpu_probs, cpu_target, cpu_probs_lengths, cpu_target_lengths,
                   Reduction::mean);
  std::tuple<Tensor, Tensor> cuda_res =
      ctc_loss_fwd(cuda_probs, cuda_target, cuda_probs_lengths,
                   cuda_target_lengths, Reduction::mean);
  Tensor cpu_loss = std::get<0>(cpu_res);
  Tensor cuda_loss = std::get<0>(cuda_res);
  Tensor cpu_log_alphas = std::get<1>(cpu_res);
  Tensor cuda_log_alphas = std::get<1>(cuda_res);
  // data compare(cpu, cuda)
  auto size_cpu_output = cpu_loss.size();
  auto size_cuda_output = cuda_loss.size();
  Tensor cuda_output_host = cuda_loss.to(kCPU);
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  for (int i = 0; i < size_cpu_output; ++i) {
    EXPECT_LE(std::abs(cpu_loss.data<scalar_t>()[i] - cuda_output_host.data<scalar_t>()[i]), 1e-1);
  }
  // bwd
  Tensor cpu_grad_probs =
      ctc_loss_bwd(cpu_probs, cpu_target, cpu_probs_lengths, cpu_target_lengths,
                   Reduction::mean, cpu_log_alphas, cpu_grad_loss);
  Tensor cuda_grad_probs = ctc_loss_bwd(
      cuda_probs, cuda_target, cuda_probs_lengths, cuda_target_lengths,
      Reduction::mean, cuda_log_alphas, cuda_grad_loss);
  // data compare(cpu, cuda)
  size_cpu_output = cpu_grad_probs.size();
  size_cuda_output = cuda_grad_probs.size();
  cuda_output_host = cuda_grad_probs.to(kCPU);
  EXPECT_EQ(size_cpu_output, size_cuda_output);
  for (int i = 0; i < size_cpu_output; ++i) {
    EXPECT_LE(
        std::abs(cpu_grad_probs.data<scalar_t>()[i] - cuda_output_host.data<scalar_t>()[i]),
        1e-1);
  }
}

INSTANTIATE_TEST_CASE_P(CTCLossInplaceTestDoubleSuite, CTCLossInplaceTestDouble,
                        ::testing::Values(CTCLossInplaceTestParamsDouble{
                            3, /* batch_size */
                            5, /* max_time */
                            4, /* max_length */
                            5  /* n_classes */
                        },
                        CTCLossInplaceTestParamsDouble{10, 20, 10, 26},
                        CTCLossInplaceTestParamsDouble{100, 50, 30, 200}
));
// #endif

#if 0
TEST(CTCLossInplaceTestDouble, FWD_None) {
  int64_t batch_size = 3;
  int64_t n_class = 5;
  int64_t max_time = 5;
  int64_t max_length = 4;

  Tensor cpu_probs({max_time, batch_size, n_class}, device(kCPU).dtype(kDataType));
  Tensor cpu_target({batch_size, max_length}, device(kCPU).dtype(kInt32));
  Tensor cpu_probs_lengths({batch_size}, device(kCPU).dtype(kInt32));
  Tensor cpu_target_lengths({batch_size}, device(kCPU).dtype(kInt32));
  Tensor cpu_grad_loss({batch_size}, device(kCPU).dtype(kInt32));

  scalar_t p_data[cpu_probs.size()] = {0.2852, 0.2179, 0.8713, 0.8132, 0.2402, 0.5268, 0.3475, 0.8771, 0.4038,
        0.0688, 0.4864, 0.1235, 0.0292, 0.3189, 0.7253, 0.1481, 0.9612, 0.3845,
        0.7857, 0.7753, 0.3206, 0.4765, 0.7693, 0.5024, 0.5571, 0.5145, 0.3191,
        0.8308, 0.5786, 0.7860, 0.2053, 0.6725, 0.2671, 0.2034, 0.4282, 0.0522,
        0.7631, 0.2774, 0.8480, 0.7933, 0.1350, 0.3623, 0.3223, 0.5438, 0.8800,
        0.5552, 0.8072, 0.1752, 0.4837, 0.9000, 0.1241, 0.3603, 0.3554, 0.9176,
        0.4854, 0.8820, 0.7800, 0.2625, 0.2821, 0.8065, 0.5258, 0.7295, 0.7651,
        0.7291, 0.5739, 0.0424, 0.3079, 0.1911, 0.1541, 0.6086, 0.1192, 0.6843,
        0.4469, 0.5021, 0.6884};
  int32_t t_data[cpu_target.size()] = {2, 1, 2, 1, 4, 4, 3, 1, 4, 1, 4, 4};
  int32_t p_l_data[cpu_probs_lengths.size()] = {5, 5, 5};
  int32_t t_l_data[cpu_target_lengths.size()] = {4,4,4};
  scalar_t g_l_data[cpu_grad_loss.size()] = {1, 1, 1};

  std::memcpy(cpu_probs.mutable_data<scalar_t>(), p_data,
              cpu_probs.size() * sizeof(scalar_t));
  std::memcpy(cpu_target.mutable_data<int32_t>(), t_data,
              cpu_target.size() * sizeof(int32_t));
  std::memcpy(cpu_probs_lengths.mutable_data<int32_t>(), p_l_data,
              cpu_probs_lengths.size() * sizeof(int32_t));
  std::memcpy(cpu_target_lengths.mutable_data<int32_t>(), t_l_data,
              cpu_target_lengths.size() * sizeof(int32_t));
  std::memcpy(cpu_grad_loss.mutable_data<scalar_t>(), g_l_data,
              cpu_grad_loss.size() * sizeof(scalar_t));

  std::tuple<Tensor, Tensor> res =
      ctc_loss_fwd(cpu_probs, cpu_target, cpu_probs_lengths, cpu_target_lengths,
                   Reduction::none);
  Tensor loss = std::get<0>(res);
  Tensor log_alphas = std::get<1>(res);

  Tensor grad_probs =
      ctc_loss_bwd(cpu_probs, cpu_target, cpu_probs_lengths,
      cpu_target_lengths, Reduction::none, log_alphas, cpu_grad_loss);

  Tensor cuda_probs = cpu_probs.to(kCUDA);
  Tensor cuda_target = cpu_target.to(kCUDA);
  Tensor cuda_probs_lengths = cpu_probs_lengths.to(kCUDA);
  Tensor cuda_target_lengths = cpu_target_lengths.to(kCUDA);
  Tensor cuda_log_alphas = log_alphas.to(kCUDA);
  Tensor cuda_grad_loss = cpu_grad_loss.to(kCUDA);

  std::tuple<Tensor, Tensor> cuda_res =
      ctc_loss_fwd(cuda_probs, cuda_target, cuda_probs_lengths, cuda_target_lengths,
                   Reduction::none);

  Tensor cuda_grad_probs =
      ctc_loss_bwd(cuda_probs, cuda_target, cuda_probs_lengths, cuda_target_lengths,
                   Reduction::none, cuda_log_alphas, cuda_grad_loss);

  TensorPrinter tp;
  std::cout << std::endl << "cpu_loss=" << std::endl;
  tp.print(loss);
  std::cout << std::endl << "cuda_grad_probs=" << std::endl;
  tp.print(cuda_grad_probs);
}
#endif

}  // namespace hice
