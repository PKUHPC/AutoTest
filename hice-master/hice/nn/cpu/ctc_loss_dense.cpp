// #include "hice/core/tensor_printer.h"
#include "hice/math/cpu/openmp/parallel.h"
#include "hice/nn/ctc_loss.h"

namespace hice {

namespace {

const int64_t kBlank = 0;

// get target prime
template <typename target_t>
static inline target_t targetPrime(const target_t *tgt, int64_t s) {
  return s % 2 == 0 ? kBlank : tgt[s / 2];
}

// log(exp(a) + exp(b))
template <typename scalar_t>
static inline scalar_t logPlusExp(scalar_t log_a1, scalar_t log_a2) {
  const scalar_t kNegInf = -std::numeric_limits<scalar_t>::infinity();
  if (log_a1 == kNegInf) {
    return log_a2;
  } else if (log_a2 == kNegInf) {
    return log_a1;
  } else {
    return (log_a1 > log_a2) ? log_a1 + log1pf(expf(log_a2 - log_a1))
                             : log_a2 + log1pf(expf(log_a1 - log_a2));
  }
};

// index help function
static inline int64_t idx(int64_t t, int64_t s, int64_t stride) {
  return stride * t + s;
}

template <typename scalar_t, typename target_t>
void ctc_loss_fwd_kernel(const Tensor &probs, const Tensor &target,
                         const Tensor &probs_lengths,
                         const Tensor &target_lengths, Tensor &loss,
                         Tensor &log_alphas) {
  const scalar_t kNegInf = -std::numeric_limits<scalar_t>::infinity();
  log_alphas.fill(kNegInf);

  const scalar_t *probs_data = probs.data<scalar_t>();
  const target_t *target_data = target.data<target_t>();
  const target_t *probs_lengths_data = probs_lengths.data<target_t>();
  const target_t *target_lengths_data = target_lengths.data<target_t>();
  scalar_t *loss_data = loss.mutable_data<scalar_t>();
  scalar_t *log_alpha_data = log_alphas.mutable_data<scalar_t>();

  int64_t batch_size = probs.dim(1);
  int64_t num_classes = probs.dim(2);  // include the "blank"=0
  int64_t max_target_length = target.dim(1);
  int64_t alpha_size = log_alphas.dim(1) * log_alphas.dim(2);

  // stride for log_alpha and probs
  int64_t las = max_target_length * 2 + 1;
  int64_t ps = probs.dim(1) * probs.dim(2);

  parallel_for(
      0, batch_size, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int i = begin; i < end; ++i) {
          const scalar_t *p_data = probs_data + i * num_classes;
          const target_t *t_data = target_data + i * max_target_length;
          scalar_t *l_data = loss_data + i;
          scalar_t *log_a_data = log_alpha_data + i * alpha_size;
          target_t target_length = target_lengths_data[i];
          target_t probs_length = probs_lengths_data[i];
          HICE_CHECK_GE(probs_length, target_length);

          // init alpha(0, 0) and alpha(0, 1)
          if (probs_length > 0) {
            log_a_data[idx(0, 0, las)] = std::log(p_data[idx(0, kBlank, ps)]);
          }
          if (target_length > 0) {
            log_a_data[idx(0, 1, las)] =
                std::log(p_data[idx(0, targetPrime(t_data, 1), ps)]);
          }

          // calculate alpha(t, s)
          for (int64_t t = 1; t < probs_length; ++t) {
            for (int64_t s = 0; s < 2 * target_length + 1; ++s) {
              target_t tps = targetPrime(t_data, s);
              scalar_t alpha_ = log_a_data[idx(t - 1, s, las)];
              if (s - 1 >= 0) {
                alpha_ = logPlusExp(alpha_, log_a_data[idx(t - 1, s - 1, las)]);
              }
              if (tps != kBlank &&
                  (s - 2 >= 0 && tps != targetPrime(t_data, s - 2))) {
                alpha_ = logPlusExp(alpha_, log_a_data[idx(t - 1, s - 2, las)]);
              }
              alpha_ += std::log(p_data[idx(t, tps, ps)]);
              log_a_data[idx(t, s, las)] = alpha_;
            }  // loop of s
          }    // loop of t

          // calculate loss
          if (target_length > 0) {
            *l_data = -logPlusExp(
                log_a_data[idx(probs_length - 1, 2 * target_length, las)],
                log_a_data[idx(probs_length - 1, 2 * target_length - 1, las)]);
          } else if (probs_length > 0) {
            *l_data = -log_a_data[idx(probs_length - 1, 0, las)];
          } else {  // probs_length=0
            *l_data = 0;
          }
        } // loop of batch i
      }); // end of parallel_for
}  // end of kernel

void ctc_loss_fwd_impl(const Tensor &probs, const Tensor &target,
                       const Tensor &probs_lengths,
                       const Tensor &target_lengths, Tensor &loss,
                       Tensor &log_alphas) {
  if (probs.scalar_type() == kFloat && target.scalar_type() == kInt32) {
    ctc_loss_fwd_kernel<float, int32_t>(probs, target, probs_lengths,
                                        target_lengths, loss, log_alphas);
  } else if (probs.scalar_type() == kFloat && target.scalar_type() == kInt64) {
    ctc_loss_fwd_kernel<float, int64_t>(probs, target, probs_lengths,
                                        target_lengths, loss, log_alphas);
  } else if (probs.scalar_type() == kDouble && target.scalar_type() == kInt32) {
    ctc_loss_fwd_kernel<double, int32_t>(probs, target, probs_lengths,
                                         target_lengths, loss, log_alphas);
  } else if (probs.scalar_type() == kDouble && target.scalar_type() == kInt64) {
    ctc_loss_fwd_kernel<double, int64_t>(probs, target, probs_lengths,
                                         target_lengths, loss, log_alphas);
  } else {
    HICE_LOG(ERROR) << "Not supported data type, probs should be float or "
                       "double, target should be int or long";
  }
}

// Backward
template <typename scalar_t, typename target_t>
void ctc_loss_bwd_kernel(const Tensor &probs, const Tensor &target,
                         const Tensor &probs_lengths,
                         const Tensor &target_lengths, Reduction reduction,
                         const Tensor &log_alphas, const Tensor &grad_loss,
                         Tensor &grad_probs) {
  const scalar_t kNegInf = -std::numeric_limits<scalar_t>::infinity();
  Tensor log_betas = full(log_alphas.dims(), kNegInf, log_alphas.options());
  grad_probs.fill(0);

  const scalar_t *probs_data = probs.data<scalar_t>();
  const target_t *target_data = target.data<target_t>();
  const target_t *probs_lengths_data = probs_lengths.data<target_t>();
  const target_t *target_lengths_data = target_lengths.data<target_t>();
  const scalar_t *log_alpha_data = log_alphas.data<scalar_t>();
  const scalar_t *grad_loss_data = grad_loss.data<scalar_t>();
  scalar_t *log_beta_data = log_betas.mutable_data<scalar_t>();
  scalar_t *grad_probs_data = grad_probs.mutable_data<scalar_t>();

  int64_t batch_size = probs.dim(1);
  int64_t num_classes = probs.dim(2);  // include the "blank"=0
  int64_t max_target_length = target.dim(1);
  int64_t alpha_size = log_alphas.dim(1) * log_alphas.dim(2);
  int64_t beta_size = alpha_size;

  // stride for log_alpha, log_beta, probs and grad_probs
  int64_t las = max_target_length * 2 + 1;
  int64_t lbs = las;
  int64_t ps = probs.dim(1) * probs.dim(2);
  int64_t gps = ps;

  parallel_for(0, batch_size, hice::GRAIN_SIZE, 
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          const scalar_t *p_data = probs_data + i * num_classes;
          const target_t *t_data = target_data + i * max_target_length;
          // do broadcast if reduced
          const scalar_t *gl_data = reduction == Reduction::none
                                        ? grad_loss_data + i
                                        : grad_loss_data;
          const scalar_t *log_a_data = log_alpha_data + i * alpha_size;
          scalar_t *log_b_data = log_beta_data + i * beta_size;
          scalar_t *gp_data = grad_probs_data + i * num_classes;
          target_t target_length = target_lengths_data[i];
          target_t probs_length = probs_lengths_data[i];
          HICE_CHECK_GE(probs_length, target_length);

          // init beta(T, S) and beta(T, S-1)
          int64_t T = probs_length - 1;
          int64_t S = 2 * target_length;
          if (T >= 0) {  // probs_length >= 1
            log_b_data[idx(T, S, lbs)] = std::log(p_data[idx(T, kBlank, ps)]);
          }
          if (S - 1 > 0) {  // target_length > 0
            log_b_data[idx(T, S - 1, lbs)] =
                std::log(p_data[idx(T, targetPrime(t_data, S - 1), ps)]);
          }

          // calculate beta(t, s)
          for (int64_t t = T - 1; t >= 0; --t) {
            for (int64_t s = S; s >= 0; --s) {
              target_t tps = targetPrime(t_data, s);
              scalar_t beta_ = log_b_data[idx(t + 1, s, lbs)];
              if (s + 1 <= S) {
                beta_ = logPlusExp(beta_, log_b_data[idx(t + 1, s + 1, lbs)]);
              }
              if (tps != kBlank &&
                  (s + 2 <= S && tps != targetPrime(t_data, s + 2))) {
                beta_ = logPlusExp(beta_, log_b_data[idx(t + 1, s + 2, lbs)]);
              }
              beta_ += std::log(p_data[idx(t, tps, ps)]);
              log_b_data[idx(t, s, las)] = beta_;
            }
          }

          // calculate prob(target | input) of sample i
          scalar_t log_p_i = kNegInf;
          if (target_length > 0) {
            log_p_i = logPlusExp(
                log_a_data[idx(probs_length - 1, 2 * target_length, las)],
                log_a_data[idx(probs_length - 1, 2 * target_length - 1, las)]);
          } else if (probs_length > 0) {
            log_p_i = log_a_data[idx(probs_length - 1, 0, las)];
          }

          // calculate grad_probs
          for (int64_t t = 0; t < probs_length; ++t) {
            // log_sum_a_b = log(sum(alpha * beta))
            std::vector<scalar_t> log_sum_ab(num_classes);
            std::fill(log_sum_ab.begin(), log_sum_ab.end(), kNegInf);
            for (int64_t s = 0; s < 2 * target_length + 1; ++s) {
              target_t tp_ = targetPrime(t_data, s);
              // for each s in {s, targetPrime(s) == k}
              log_sum_ab[tp_] =
                  logPlusExp(log_sum_ab[tp_], log_a_data[idx(t, s, las)] +
                                                  log_b_data[idx(t, s, lbs)]);
            }  // loop of s

            // The grad_probs has stored value log(sum(alpha * beta)),
            // now it is going to calculate the final grad probs
            for (int64_t k = 0; k < num_classes; ++k) {
              // y(t, k) = log_ytk
              scalar_t log_ytk = std::log(p_data[idx(t, k, ps)]);
              gp_data[idx(t, k, gps)] =
                  (-std::exp(log_sum_ab[k] - log_p_i - 2 * log_ytk)) *
                  (*gl_data);
            }  // loop of k
          }    // loop of t

          // if (i == 0) {
          //   scalar_t a11 = std::exp(log_a_data[idx(1, 1, las)]);
          //   scalar_t b11 = std::exp(log_b_data[idx(1, 1, lbs)]);
          //   scalar_t a15 = std::exp(log_a_data[idx(1, 5, las)]);
          //   scalar_t b15 = std::exp(log_b_data[idx(1, 5, lbs)]);
          //   scalar_t a47 = std::exp(log_a_data[idx(4, 7, las)]);
          //   scalar_t a48 = std::exp(log_a_data[idx(4, 8, las)]);
          //   scalar_t y12 = p_data[idx(1, 2, ps)];
          //   scalar_t g_ = -(a11 * b11 + a15 * b15) / (a47 + a48) / (y12 *
          //   y12); std::cout << "a11=" << log_a_data[idx(1, 1, las)] <<
          //   std::endl; std::cout << "b11=" << log_b_data[idx(1, 1, lbs)] <<
          //   std::endl; std::cout << "a15=" << log_a_data[idx(1, 5, las)] <<
          //   std::endl; std::cout << "b15=" << log_b_data[idx(1, 5, lbs)] <<
          //   std::endl; std::cout << "a47=" << log_a_data[idx(4, 7, las)] <<
          //   std::endl; std::cout << "a48=" << log_a_data[idx(4, 8, las)] <<
          //   std::endl; std::cout << "y12=" << p_data[idx(1, 2, ps)] <<
          //   std::endl; std::cout << "grad_12=" << g_ << std::endl;
          // }

        }  // loop of batch_size i
      }); // end of parallel_for
  // TensorPrinter tp;
  // std::cout << std::endl << "log_alphas=" << std::endl;
  // tp.print(log_alphas);
  // std::cout << std::endl << "log_betas=" << std::endl;
  // tp.print(log_betas);
}  // end of kernel

void ctc_loss_bwd_impl(const Tensor &probs, const Tensor &target,
                       const Tensor &probs_lengths,
                       const Tensor &target_lengths, Reduction reduction,
                       const Tensor &log_alphas, const Tensor &grad_loss,
                       Tensor &grad_probs) {
  if (probs.scalar_type() == kFloat && target.scalar_type() == kInt32) {
    ctc_loss_bwd_kernel<float, int32_t>(probs, target, probs_lengths,
                                        target_lengths, reduction, log_alphas,
                                        grad_loss, grad_probs);
  } else if (probs.scalar_type() == kFloat && target.scalar_type() == kInt64) {
    ctc_loss_bwd_kernel<float, int64_t>(probs, target, probs_lengths,
                                        target_lengths, reduction, log_alphas,
                                        grad_loss, grad_probs);
  } else if (probs.scalar_type() == kDouble && target.scalar_type() == kInt32) {
    ctc_loss_bwd_kernel<double, int32_t>(probs, target, probs_lengths,
                                         target_lengths, reduction, log_alphas,
                                         grad_loss, grad_probs);
  } else if (probs.scalar_type() == kDouble && target.scalar_type() == kInt64) {
    ctc_loss_bwd_kernel<double, int64_t>(probs, target, probs_lengths,
                                         target_lengths, reduction, log_alphas,
                                         grad_loss, grad_probs);
  } else {
    HICE_LOG(ERROR) << "Not supported data type, probs should be float or "
                       "double, target should be int or long";
  }
}

}  // namespace

// Forward
HICE_REGISTER_KERNEL(ctc_loss_fwd_dispatcher, &ctc_loss_fwd_impl,
                     {kCPU, kDense},  // probs
                     {kCPU, kDense},  // target
                     {kCPU, kDense},  // probs_lengths
                     {kCPU, kDense},  // target_lengths
                     {kCPU, kDense},  // loss
                     {kCPU, kDense}   // log_alphas
);

// Backward
HICE_REGISTER_KERNEL(ctc_loss_bwd_dispatcher, &ctc_loss_bwd_impl,
                     {kCPU, kDense},  // probs
                     {kCPU, kDense},  // target
                     {kCPU, kDense},  // probs_lengths
                     {kCPU, kDense},  // target_lengths
                     {kCPU, kDense},  // log_alphas
                     {kCPU, kDense},  // grad_loss
                     {kCPU, kDense}   // grad_probs
);

}  // namespace hice
