#include "attention.h"
#include <math.h>
#include <stdio.h>
#include "float.h"
#include "src/basic/index_utils.h"
#include "src/core/allocator.h"
#include "src/core/utils.h"
#define max(a, b) ((a > b) ? a : b)

/* The implementation of attention for the float data type */
static Status aitisa_attention_float(const Tensor query, const Tensor key,
                                     const Tensor value, const int is_causal,
                                     Tensor* output) {

  int64_t batch = aitisa_tensor_dim(query, 0);
  int64_t seq_q = aitisa_tensor_dim(query, 1);
  int64_t head = aitisa_tensor_dim(query, 2);
  int64_t dim = aitisa_tensor_dim(query, 3);
  int64_t seq_k = aitisa_tensor_dim(key, 1);

  int64_t ndim = aitisa_tensor_ndim(query);

  int64_t* out_dims;
  int64_t out_dims_vector[4] = {batch, seq_q, head, dim};
  out_dims = out_dims_vector;

  int64_t* query_strides =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*query_strides) * ndim);
  int64_t* key_strides =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*key_strides) * ndim);
  int64_t* value_strides =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*value_strides) * ndim);
  int64_t* out_strides =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*out_strides) * ndim);
  Tensor out_tensor;
  DataType dtype = aitisa_tensor_data_type(query);
  Device device = aitisa_tensor_device(query);
  CHECK_STATUS(
      aitisa_create(dtype, device, out_dims, ndim, NULL, 0, &out_tensor));
  *output = out_tensor;

  aitisa_get_all_strides(query, query_strides);
  aitisa_get_all_strides(key, key_strides);
  aitisa_get_all_strides(value, value_strides);
  aitisa_get_all_strides(*output, out_strides);

  const size_t row_shift = seq_k - seq_q;

  float* q_ptr = aitisa_tensor_data(query);
  float* k_ptr = aitisa_tensor_data(key);
  float* v_ptr = aitisa_tensor_data(value);
  float* o_ptr = aitisa_tensor_data(*output);
  // S = Q * K^T

  int64_t* s_dims =
      aitisa_default_cpu_allocator()->raw_alloc(sizeof(*s_dims) * ndim);
  int64_t s_dims_vector[4] = {batch, seq_q, head, seq_k};
  s_dims = s_dims_vector;
  Tensor s_tensor;

  CHECK_STATUS(aitisa_create(dtype, device, s_dims, ndim, NULL, 0, &s_tensor));
  Tensor* s = &s_tensor;

  float* s_ptr = aitisa_tensor_data(*s);

  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < head; ++h) {
      for (size_t sq = 0; sq < seq_q; ++sq) {
        for (size_t sk = 0; sk < seq_k; ++sk) {
          float tmp = 0.0;
          for (size_t d = 0; d < dim; ++d) {
            tmp += q_ptr[b * (seq_q * head * dim) + sq * (head * dim) +
                         h * dim + d] *
                   k_ptr[b * (seq_k * head * dim) + sk * (head * dim) +
                         h * dim + d];
          }
          s_ptr[b * (seq_q * head * seq_k) + sq * (head * seq_k) + h * seq_k +
                sk] = tmp;
        }
      }
    }
  }
  // P = Softmax(S)
  int64_t* p_dims;
  int64_t p_dims_vector[4] = {batch, seq_q, head, seq_k};
  p_dims = p_dims_vector;
  Tensor p_tensor;
  CHECK_STATUS(aitisa_create(dtype, device, p_dims, ndim, NULL, 0, &p_tensor));
  Tensor* p = &p_tensor;

  float* p_ptr = aitisa_tensor_data(*p);

  float scale = 1.0 / sqrt(dim);
  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < head; ++h) {
      for (size_t sq = 0; sq < seq_q; ++sq) {
        size_t row = seq_q;
        if (is_causal) {
          row = sq + row_shift + 1;
        }

        // Max(S)
        float* tmp_s =
            aitisa_default_cpu_allocator()->raw_alloc(sizeof(*tmp_s) * seq_k);

        float max_s = -FLT_MIN;
        for (size_t sk = 0; sk < row; ++sk) {
          tmp_s[sk] = s_ptr[b * (seq_q * head * seq_k) + sq * (head * seq_k) +
                            h * seq_k + sk] *
                      scale;
          max_s = max(max_s, tmp_s[sk]);
        }
        // Sum(S)
        float sum_s = 0.0;
        for (size_t sk = 0; sk < row; ++sk) {
          tmp_s[sk] = exp(tmp_s[sk] - max_s);
          sum_s += tmp_s[sk];
        }

        // Softmax(S)
        for (size_t sk = 0; sk < row; ++sk) {
          p_ptr[b * (seq_q * head * seq_k) + sq * (head * seq_k) + h * seq_k +
                sk] = tmp_s[sk] / sum_s;
        }

        // Causal(S)
        if (is_causal) {
          for (size_t sk = row; sk < seq_q; ++sk) {
            p_ptr[b * (seq_q * head * seq_k) + sq * (head * seq_k) + h * seq_k +
                  sk] = 0.0;
          }
        }
        aitisa_default_cpu_allocator()->raw_dealloc(tmp_s);
      }
    }
  }

  // O = P * V
  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < head; ++h) {
      for (size_t sq = 0; sq < seq_q; ++sq) {
        for (size_t d = 0; d < dim; ++d) {
          float tmp = 0.0;
          for (size_t sk = 0; sk < seq_k; ++sk) {
            tmp += p_ptr[b * (seq_q * head * seq_k) + sq * (head * seq_k) +
                         h * seq_k + sk] *
                   v_ptr[b * (seq_k * head * dim) + sk * (head * dim) +
                         h * dim + d];
          }
          o_ptr[b * (seq_q * head * dim) + sq * (head * dim) + h * dim + d] =
              tmp;
        }
      }
    }
  }
  aitisa_default_cpu_allocator()->raw_dealloc(query_strides);
  aitisa_default_cpu_allocator()->raw_dealloc(key_strides);
  aitisa_default_cpu_allocator()->raw_dealloc(value_strides);
  aitisa_default_cpu_allocator()->raw_dealloc(out_strides);
}

Status aitisa_attention(const Tensor query, const Tensor key,
                        const Tensor value, const int is_causal,
                        Tensor* output) {
  DataType dtype = aitisa_tensor_data_type(query);
  switch (dtype.code) {
    case TYPE_FLOAT:
      CHECK_STATUS(
          aitisa_attention_float(query, key, value, is_causal, output));
      break;
    default:
      return STATUS_NOT_SUPPORTED;
  }
}
