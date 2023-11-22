#ifndef AITISA_API_ATTENTION_H
#define AITISA_API_ATTENTION_H

#include "src/core/tensor.h"

/**
 * @brief attention.
 *
 * @param Query The input tensor of a Query [m_batch, m_seq_q, m_head, m_dim].
 *
 * @param Key The input tensor of a shape [m_batch, m_seq_k, m_head, m_dim].
 *
 * @param Value The filter tensor of a shape [m_batch, m_seq_k, m_head, m_dim].
 *
 * @param Output The filter tensor of a shape [m_batch, m_seq_q, m_head, m_dim].
 *
 * @param is_causal is_causal is a hint that the mask is causal.
 *
 * @return Status The Status enum indicates whether the routine is OK.
 */

AITISA_API_PUBLIC Status aitisa_attention(const Tensor query, const Tensor key, const Tensor value, const int is_causal,
                                          Tensor *output);


#endif  //AITISA_API_ATTENTION_HW
