#include "hice/basic/factories.h"
#include "hice/core/tensor_printer.h"
#include "hice/math/matmul.h"

using namespace hice;
int main() {
  TensorPrinter tp;

  // CPU matmul 
  std::cout << "==============================" << std::endl;
  std::cout << "     CPU matmul example       " << std::endl;
  std::cout << "==============================" << std::endl;
  Tensor h_mat1 = full({4, 4}, 1, device(kCPU).dtype(kDouble));
  Tensor h_mat2 = full({4, 4}, 1, device(kCPU).dtype(kDouble));
  Tensor h_mat3 = matmul(h_mat1, h_mat2);
  tp.print(h_mat3);

  // CUDA matmul 
  std::cout << "==============================" << std::endl;
  std::cout << "     CUDA matmul example      " << std::endl;
  std::cout << "==============================" << std::endl;
  Tensor d_mat1 = full({4, 4}, 1, device(kCUDA).dtype(kDouble));
  Tensor d_mat2 = full({4, 4}, 1, device(kCUDA).dtype(kDouble));
  Tensor d_mat3 = matmul(d_mat1,d_mat2);
  tp.print(d_mat3);
}
