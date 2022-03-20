#include "hice/math/matmul.h"
#include "hice/core/sparse_tensor.h"

namespace hice {

HICE_DEFINE_DISPATCHER(matmul_dispatcher);

// outplace
Tensor matmul(const Tensor& tensor_a, const Tensor& tensor_b,
              MatmulOption option_a, MatmulOption option_b) {
  // std::cout<<"In matmul:"<<std::endl;
  if (tensor_a.is_coo() && tensor_b.is_coo()) {
    Tensor result = new_sparse(device(tensor_a.device()).dtype(tensor_a.data_type()));
    matmul_dispatcher(tensor_a, tensor_b, result, option_a, option_b,
                      /* resizable = */ true);
    return result;
  } else if (tensor_a.is_csr() && tensor_b.is_csr()) {
    Tensor result = new_csr(device(tensor_a.device()).dtype(tensor_a.data_type()));
    matmul_dispatcher(tensor_a, tensor_b, result, option_a, option_b,
                      /* resizable = */ true);
    return result;
  } else {
    Tensor result({}, device(tensor_a.device()).dtype(tensor_a.data_type()));
    matmul_dispatcher(tensor_a, tensor_b, result, option_a, option_b,
                      /* resizable = */ true);
    return result;
  }
}

// inplace
Tensor& matmul(const Tensor& tensor_a, const Tensor& tensor_b, Tensor& result,
               MatmulOption option_a, MatmulOption option_b) {
  // std::cout<<"hice matmul fwd begin"<<std::endl;
  HICE_CHECK_ARGUMENT(!result.is_same(tensor_a));
  HICE_CHECK_ARGUMENT(!result.is_same(tensor_b));
  HICE_CHECK_ARGUMENT(result.is_dense()) << "result must be dense for matmul_out";
  matmul_dispatcher(tensor_a, tensor_b, result, option_a, option_b,
                    /* resizable = */ false);
  return result;
}

}  // namespace hice
