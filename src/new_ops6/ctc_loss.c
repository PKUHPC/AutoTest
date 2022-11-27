#include "src/new_ops6/ctc_loss.h"
#include <float.h>
#include <math.h>
#include "src/basic/factories.h"
#include "src/core/allocator.h"
#include "src/core/dispatch.h"

const int64_t kBlank = 0;

// index help function
static inline int64_t idx(int64_t t, int64_t s, int64_t stride) {
  return stride * t + s;
}
static inline int64_t targetPrime(const int64_t* tgt, int64_t s) {
  return s % 2 == 0 ? kBlank : tgt[s / 2];
}
static Status ctc_loss_create_output(const Tensor input, Tensor* output) {
  Status status;
  int64_t* dims = aitisa_tensor_dims(input);
  Tensor new_tensor;
  DataType dtype = aitisa_tensor_data_type(input);
  Device device = aitisa_tensor_device(input);
  int64_t new_tensor_dims[1] = {dims[1]};
  status = aitisa_full(dtype, device, new_tensor_dims, 1, 0, &new_tensor);
  *output = new_tensor;

  return status;
}

static double logPlusExp(double log_a1, double log_a2) {
  if (log_a1 == -DBL_MAX) {
    return log_a2;
  } else if (log_a2 == -DBL_MAX) {
    return log_a1;
  } else {
    return (log_a1 > log_a2) ? log_a1 + log1pf(expf(log_a2 - log_a1))
                             : log_a2 + log1pf(expf(log_a1 - log_a2));
  }
};

#define ctc_cross_kernel(typename)                                             \
  typename* probs_data = aitisa_tensor_data(probs);                            \
  int64_t* target_data = (int64_t*)aitisa_tensor_data(target);                 \
  int64_t* probs_lengths_data = (int64_t*)aitisa_tensor_data(probs_lengths);   \
  int64_t* target_lengths_data = (int64_t*)aitisa_tensor_data(target_lengths); \
                                                                               \
  typename* loss_data = aitisa_tensor_data(*loss);                             \
                                                                               \
  Tensor log_alphas;                                                           \
  int64_t log_alphas_dims[3] = {aitisa_tensor_dims(probs)[1],                  \
                                aitisa_tensor_dims(probs)[0],                  \
                                aitisa_tensor_dims(target)[1] * 2 + 1};        \
  if (aitisa_tensor_data_type(probs).code == 8) {                              \
    status = aitisa_full(aitisa_tensor_data_type(probs),                       \
                         aitisa_tensor_device(probs), log_alphas_dims, 3,      \
                         -FLT_MAX, &log_alphas);                               \
  } else if (aitisa_tensor_data_type(probs).code == 9) {                       \
    status = aitisa_full(aitisa_tensor_data_type(probs),                       \
                         aitisa_tensor_device(probs), log_alphas_dims, 3,      \
                         -DBL_MAX, &log_alphas);                               \
  }                                                                            \
                                                                               \
  typename* log_alphas_data = aitisa_tensor_data(log_alphas);                  \
                                                                               \
  int64_t batch_size = aitisa_tensor_dims(probs)[1];                           \
  int64_t num_classes = aitisa_tensor_dims(probs)[2];                          \
  int64_t max_target_length = aitisa_tensor_dims(target)[1];                   \
  int64_t alpha_size =                                                         \
      aitisa_tensor_dims(log_alphas)[1] * aitisa_tensor_dims(log_alphas)[2];   \
                                                                               \
  int64_t las = max_target_length * 2 + 1;                                     \
  int64_t ps = aitisa_tensor_dims(probs)[1] * aitisa_tensor_dims(probs)[2];    \
                                                                               \
  for (int i = 0; i < batch_size; ++i) {                                       \
    const typename* p_data = probs_data + i * num_classes;                     \
    const int64_t* t_data = target_data + i * max_target_length;               \
    typename* l_data = loss_data + i;                                          \
    typename* log_a_data = log_alphas_data + i * alpha_size;                   \
    int64_t target_length = target_lengths_data[i];                            \
    int64_t probs_length = probs_lengths_data[i];                              \
    if (probs_length > 0) {                                                    \
      log_a_data[idx(0, 0, las)] = log(p_data[idx(0, kBlank, ps)]);            \
    }                                                                          \
    if (target_length > 0) {                                                   \
      log_a_data[idx(0, 1, las)] =                                             \
          log(p_data[idx(0, targetPrime(t_data, 1), ps)]);                     \
    }                                                                          \
    for (int64_t t = 1; t < probs_length; ++t) {                               \
      for (int64_t s = 0; s < 2 * target_length + 1; ++s) {                    \
        int64_t tps = targetPrime(t_data, s);                                  \
        typename alpha_ = log_a_data[idx(t - 1, s, las)];                      \
        if (s - 1 >= 0) {                                                      \
          alpha_ = logPlusExp(alpha_, log_a_data[idx(t - 1, s - 1, las)]);     \
        }                                                                      \
        if (tps != kBlank &&                                                   \
            (s - 2 >= 0 && tps != targetPrime(t_data, s - 2))) {               \
          alpha_ = logPlusExp(alpha_, log_a_data[idx(t - 1, s - 2, las)]);     \
        }                                                                      \
        alpha_ += log(p_data[idx(t, tps, ps)]);                                \
        log_a_data[idx(t, s, las)] = alpha_;                                   \
      }                                                                        \
    }                                                                          \
    if (target_length > 0) {                                                   \
      *l_data = -logPlusExp(                                                   \
          log_a_data[idx(probs_length - 1, 2 * target_length, las)],           \
          log_a_data[idx(probs_length - 1, 2 * target_length - 1, las)]);      \
    } else if (probs_length > 0) {                                             \
      *l_data = -log_a_data[idx(probs_length - 1, 0, las)];                    \
    } else {                                                                   \
      *l_data = 0;                                                             \
    }                                                                          \
  }

Status aitisa_ctc_loss(const Tensor probs, const Tensor target,
                       const Tensor probs_lengths, const Tensor target_lengths,
                       Tensor* loss) {

  Status status = STATUS_SUCCESS;
  DataType prods_dtype = aitisa_tensor_data_type(probs);
  DataType target_dtype = aitisa_tensor_data_type(target);
  int64_t max_time = aitisa_tensor_dims(probs)[0];
  int64_t batch_size = aitisa_tensor_dims(probs)[1];
  int64_t max_target_length = aitisa_tensor_dims(target)[1];
  CHECK_STATUS(ctc_loss_create_output(probs, loss));

  AITISA_DISPATCH_ALL_TYPES_RETURN(prods_dtype, ctc_cross_kernel);

  return status;
}
