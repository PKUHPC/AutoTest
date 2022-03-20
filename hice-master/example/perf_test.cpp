#include <hice/math/unary_expr.h>
#include <hice/math/binary_expr.h>
#include <hice/math/matmul.h>
#include <hice/math/reduce.h>
#include <hice/nn/activation.h>
#include <hice/nn/batch_norm.h>
#include <hice/nn/conv.h>
#include <hice/nn/cross_entropy.h>
#include <hice/nn/ctc_loss.h>
#include <hice/nn/dropout.h>
#include <hice/nn/l1_loss.h>
#include <hice/nn/mse_loss.h>
#include <hice/nn/nll_loss.h>
#include <hice/nn/pooling.h>
#include <hice/nn/smooth_l1_loss.h>
#include <hice/nn/softmax_cross_entropy.h>
#include <hice/nn/sparse_softmax_cross_entropy.h>
#include <hice/nn/softmax.h>

#include <hice/util/types.h>
#include <hice/core/tensor_printer.h>

#include "benchmark.h"

#include <cmath>

using namespace hice;

// using BenchMarker = BenchmarkCPU;
// const static DeviceType kDEVICE = kCPU;

using BenchMarker = BenchmarkCUDA;
const static DeviceType kDEVICE = kCUDA;

#define BENCH_WITH_LOG(func_name, ...)  \
  { \
    float time = BenchMarker::bench([&](){ func_name(__VA_ARGS__); }); \
    std::cout << #func_name << "," << time << ",ms." << std::endl; \
  }

void unary_test() {
  float one_val = 1.0;

  std::vector<int64_t> dims_in = {32, 30, 112, 112};
  std::vector<int64_t> dims_out = {32, 30, 112, 112};

  hice::Tensor input = full(dims_in, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor output = empty(dims_out, device(kDEVICE).dtype(kFloat));
  BENCH_WITH_LOG(exp, input, output);
  BENCH_WITH_LOG(log, input, output);
  BENCH_WITH_LOG(neg, input, output);
}


void binary_test() {
  float one_val = 1.0;

  std::vector<int64_t> dims_in = {32, 30, 112, 112};
  std::vector<int64_t> dims_out = {32, 30, 112, 112};

  hice::Tensor lhs = full(dims_in, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor rhs = full(dims_in, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor output = empty(dims_out, device(kDEVICE).dtype(kFloat));
  BENCH_WITH_LOG(add, lhs, rhs, output);
  BENCH_WITH_LOG(sub, lhs, rhs, output);
  BENCH_WITH_LOG(mul, lhs, rhs, output);
  BENCH_WITH_LOG(div, lhs, rhs, output);
}


void matmul_test() {
  float one_val = 1.0;

  std::vector<int64_t> dims = {4096, 4096};

  hice::Tensor lhs = full(dims, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor rhs = full(dims, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor output = empty(dims, device(kDEVICE).dtype(kFloat));
  BENCH_WITH_LOG(matmul, lhs, rhs, output);
}

void reduce_test() {
  float one_val = 1.0;

  std::vector<int64_t> dims = {4096, 4096};
  std::vector<int64_t> reduce_dims = {0, 1};

  hice::Tensor input = full(dims, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor output = empty({1}, device(kDEVICE).dtype(kFloat));
  BENCH_WITH_LOG(reduce_mean, input, reduce_dims, false, output);
}


void activation_test() {
  float one_val = 1.0;

  std::vector<int64_t> dims_in = {32, 30, 112, 112};
  std::vector<int64_t> dims_out = {32, 30, 112, 112};

  hice::Tensor input = full(dims_in, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor output = empty(dims_out, device(kDEVICE).dtype(kFloat));
  BENCH_WITH_LOG(abs_fwd, input, output);
  BENCH_WITH_LOG(relu_fwd, input, output);
  BENCH_WITH_LOG(sigmoid_fwd, input, output);
  BENCH_WITH_LOG(sqrt_fwd, input, output);
  BENCH_WITH_LOG(square_fwd, input, output);
  BENCH_WITH_LOG(tanh_fwd, input, output);
  BENCH_WITH_LOG(elu_fwd, input, 0.1, output);
}


void batch_norm_test() {
  float one_val = 1.0, zero_val = 0.0;
  int64_t N = 32, C = 3, H = 112, W = 112;
  double eps = 1e-5, momentum = 0.1;
  std::vector<int64_t> dims = {N, C, H, W};
  std::vector<int64_t> dims_ch = {C};
  
  hice::Tensor input = full(dims, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor scale = full(dims_ch, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor bias = full(dims_ch, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor running_mean = full(dims_ch, zero_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor running_var = full(dims_ch, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor output = empty(dims, device(kDEVICE).dtype(kFloat));
  hice::Tensor saved_mean = empty(dims_ch, device(kDEVICE).dtype(kFloat));
  hice::Tensor saved_var = empty(dims_ch, device(kDEVICE).dtype(kFloat));

  BENCH_WITH_LOG(batch_norm_fwd, input, scale, bias, running_mean, running_var, 
                  true, HICE_BATCHNORM_SPATIAL, momentum, eps, 
                  output, saved_mean, saved_var);
}


void conv_test() {
  float one_val = 1.0;
  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> padding = {1, 1};
  std::vector<int64_t> dilation = {1, 1};
  std::vector<int64_t> dims_in = {32, 3, 112, 112};
  std::vector<int64_t> dims_weight = {32, 3, 3, 3};
  std::vector<int64_t> dims_out = {32, 32, 112, 112};
  
  hice::Tensor input = full(dims_in, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor weight = full(dims_weight, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor output = empty(dims_out, device(kDEVICE).dtype(kFloat));
  BENCH_WITH_LOG(conv_fwd, input, weight, hice::nullopt, padding, stride, dilation, 1, false, true, output);
}


void ctc_loss_test() {
  int64_t batch_size = 32;
  int64_t max_time = 80;
  int64_t max_length = 80;
  int64_t n_classes = 100;

  Tensor probs = rand_uniform({max_time, batch_size, 100}, 0.0, 1.0, device(kDEVICE).dtype(kFloat));
  Tensor target = full({batch_size, max_length}, 1, device(kDEVICE).dtype(kFloat)).to(kInt64);
  Tensor probs_lengths = full({batch_size}, 80, device(kDEVICE).dtype(kInt64));
  Tensor target_lengths = rand_uniform({batch_size}, 0, 80, device(kDEVICE).dtype(kFloat)).to(kInt64);
  Tensor loss = rand_uniform({batch_size}, 0.0, 10.0, device(kDEVICE).dtype(kFloat));

  BENCH_WITH_LOG(ctc_loss_fwd, probs, target, probs_lengths, target_lengths, Reduction::mean)
}


void dropout_test() {
  
  int64_t N = 32, C = 3, H = 112, W = 112;

  float one_val = 1.0, rate = 0.8;
  std::vector<int64_t> dims = {N, C, H, W};

  Tensor input = full(dims, one_val, device(kDEVICE).dtype(kFloat));
  Tensor mask = empty(dims, device(kDEVICE).dtype(kBool));
  Tensor output = empty(dims, device(kDEVICE).dtype(kFloat));

  BENCH_WITH_LOG(dropout_fwd, input, rate, mask, output);
}


void pool_test() {
  float one_val = 1.0;

  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> padding = {0, 0};
  std::vector<int64_t> kernel = {3, 3};
  std::vector<int64_t> dims_in = {32, 3, 112, 112};
  std::vector<int64_t> dims_out = {32, 3, 110, 110};

  hice::Tensor input = full(dims_in, one_val, device(kDEVICE).dtype(kFloat));
  hice::Tensor output = empty(dims_out, device(kDEVICE).dtype(kFloat));
  BENCH_WITH_LOG(pooling_avg_fwd, input, kernel, stride, padding, output);
}


void sparse_softmax_cross_entropy_test() {
  float one_val = 1.0;
  Tensor logit = rand_uniform({320, 1000}, 0, 10, dtype(kFloat).device(kDEVICE));
  Tensor target = rand_uniform({logit.dim(0)}, 0, logit.dim(1), dtype(kFloat).device(kDEVICE)).to(kInt64);
  Tensor prob = rand_uniform(logit.dims(), 0, one_val, dtype(kFloat).device(kDEVICE));
  Tensor loss = rand_uniform(target.dims(), 0, one_val, dtype(kFloat).device(kDEVICE));
  BENCH_WITH_LOG(sparse_softmax_cross_entropy_fwd, logit, target, hice::nullopt, 1, prob, loss);
}

int main() {

  unary_test();
  binary_test();
  matmul_test();
  reduce_test();
  activation_test();
  batch_norm_test();
  conv_test();
  ctc_loss_test();
  dropout_test();
  pool_test();
  sparse_softmax_cross_entropy_test();

  return 0;
}