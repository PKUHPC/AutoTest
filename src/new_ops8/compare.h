#ifndef COMPARE_H
#define COMPARE_H

#include "src/core/tensor.h"

/**
 * @brief Enumeration type for all possible compare operation types
 *
 * @details Use to select the right operator in calculation
 */
typedef enum {
  OP_EQUAL = 0U,
  OP_GREATER,
  OP_GREATER_EQUAL,
  OP_LESS,
  OP_LESS_EQUAL,
  OP_COMPARE_NOPS = 5U /**< The total number of all possible operations */
} CompareOpCode;

typedef void (*CompareOpFunc)(void *a, void *b, CompareOpCode op, void *c);
CompareOpFunc aitisa_compare_op_func(DataType dtype);

/**
 * @brief Computes \text{tensor1} = \text{tensor2}input≥other element-wise.
 *
 * @param tensor1 the tensor to compare.
 * @param tensor2 the tensor to compare.
 * @param output  the output tensor.
 *
 * @return
 * @retval STATUS_SUCCESS Successfully add two tensors
 * @retval STATUS_TYPE_MISMATCH The datatype of two tensors should be consistent
 * @retval STATUS_NOT_SUPPORTED Device is not supported
 * @retval STATUS_INVALID_ARGUMENT The dimension of two tensors is not consistent
 */
AITISA_API_PUBLIC Status aitisa_equal(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);

/**
 * @brief Computes \text{input} > \text{other}input>other element-wise.
 *
 * @param tensor1 the tensor to compare.
 * @param tensor2 the tensor to compare.
 * @param output  the output tensor.
 *
 * @return
 * @retval STATUS_SUCCESS Successfully subtract two tensors
 * @retval STATUS_TYPE_MISMATCH The datatype of two tensors should be consistent
 * @retval STATUS_NOT_SUPPORTED Device is not supported
 * @retval STATUS_INVALID_ARGUMENT The dimension of two tensors is not consistent
 */
AITISA_API_PUBLIC Status aitisa_greater(const Tensor tensor1, const Tensor tensor2,
                                        Tensor *output);

/**
 * @brief Computes \text{tensor1} \geq \text{tensor2}input≥other element-wise.
 *
 * @param tensor1 the tensor to compare.
 * @param tensor2 the tensor to compare.
 * @param output  the output tensor.
 *
 * @return
 * @retval STATUS_SUCCESS Successfully subtract two tensors
 * @retval STATUS_TYPE_MISMATCH The datatype of two tensors should be consistent
 * @retval STATUS_NOT_SUPPORTED Device is not supported
 * @retval STATUS_INVALID_ARGUMENT The dimension of two tensors is not consistent
 */
AITISA_API_PUBLIC Status aitisa_greater_equal(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);

/**
 * @brief Computes \text{input} < \text{other}input<other element-wise.
 *
 * @param tensor1 the tensor to compare.
 * @param tensor2 the tensor to compare.
 * @param output  the output tensor.
 *
 * @return
 * @retval STATUS_SUCCESS Successfully subtract two tensors
 * @retval STATUS_TYPE_MISMATCH The datatype of two tensors should be consistent
 * @retval STATUS_NOT_SUPPORTED Device is not supported
 * @retval STATUS_INVALID_ARGUMENT The dimension of two tensors is not consistent
 */
AITISA_API_PUBLIC Status aitisa_less(const Tensor tensor1, const Tensor tensor2,
                                     Tensor *output);

/**
 * @brief Computes \text{input} \leq \text{other}input≤other element-wise.
 *
 * @param tensor1 the tensor to compare.
 * @param tensor2 the tensor to compare.
 * @param output  the output tensor.
 *
 * @return
 * @retval STATUS_SUCCESS Successfully subtract two tensors
 * @retval STATUS_TYPE_MISMATCH The datatype of two tensors should be consistent
 * @retval STATUS_NOT_SUPPORTED Device is not supported
 * @retval STATUS_INVALID_ARGUMENT The dimension of two tensors is not consistent
 */
AITISA_API_PUBLIC Status aitisa_less_equal(const Tensor tensor1, const Tensor tensor2,
                                    Tensor *output);

#endif
