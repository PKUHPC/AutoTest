#include "hice/device/cuda/context_cuda.h"
#include "hice/core/shape_util.h"
#include "hice/basic/reshape.h"
#include "hice/basic/cpu/index_helper.h"
#include "hice/math/matmul.h"
#include "hice/basic/transpose.h"

namespace hice {

namespace {

void vv(const Tensor& tensor1, const Tensor& tensor2, Tensor& result) {
  CUDAContext cuda_ctx(tensor1.device());
  cublasHandle_t handle = cuda_ctx.cublas_handle();
  HICE_CUBLAS_CHECK(cublasSetPointerMode( 
    handle, CUBLAS_POINTER_MODE_DEVICE));
  const int dim_tensor1 = tensor1.dim(0);
  const int dim_tensor2 = tensor2.dim(0);
  HICE_CHECK_EQ(dim_tensor1, dim_tensor2) 
      << "Both dimensions of arguments to dot must be equal";
  const int n = dim_tensor1;
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2) 
      << "Both scalar types of arguments to dot must be equal";
  ScalarType sc_type = sc_type_tensor1; 
  const int incx = 1;
  const int incy = 1;
  switch (sc_type) {
    case ScalarType::Float: {
      using scalar_t = float;
      const scalar_t *x = tensor1.data<scalar_t>();
      const scalar_t *y = tensor2.data<scalar_t>();
      scalar_t *r = result.mutable_data<scalar_t>();
      HICE_CUBLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, r));
      break;
    }
    case ScalarType::Double: {
      using scalar_t = double;
      const scalar_t *x = tensor1.data<scalar_t>();
      const scalar_t *y = tensor2.data<scalar_t>();
      scalar_t *r = result.mutable_data<scalar_t>();
      HICE_CUBLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, r));
      break;
    }
    default:
      HICE_LOG(ERROR) << "This function doesn't handle types other than " 
                         "float, double, complex<float>, complex<double>";
  }
}

void mv(const Tensor &tensor1, const Tensor &tensor2, Tensor &result,
        MatmulOption option) {
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2) 
      << "Both scalar types of arguments to mv must be equal";
  cublasOperation_t trans = option == kNoTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUDAContext cuda_ctx(tensor1.device());
  cublasHandle_t handle = cuda_ctx.cublas_handle();
  HICE_CUBLAS_CHECK(cublasSetPointerMode( 
   handle, CUBLAS_POINTER_MODE_HOST));
  const int dim0_tensor1 = tensor1.dim(0);
  const int dim1_tensor1 = tensor1.dim(1);
  const int m = trans == CUBLAS_OP_T ? dim1_tensor1 : dim0_tensor1;
  HICE_CHECK_EQ(m, tensor2.dim(0)) 
      << "Dimensions of arguments to mv must be matched";
  ScalarType sc_type = sc_type_tensor1;
  const int lda = dim1_tensor1;
  const int incx = 1;
  const int incy = 1;
  // transpose by default because of column priority
  switch (sc_type) {
    case ScalarType::Float:{
      using scalar_t = float;
      const scalar_t *a = tensor1.data<scalar_t>();
      const scalar_t *x = tensor2.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      HICE_CUBLAS_CHECK(cublasSgemv(handle, trans, dim1_tensor1, dim0_tensor1, 
                  &alpha, a, lda, x, incx, &beta, y, incy));
      break;
    }
    case ScalarType::Double:{
      using scalar_t = double;
      const scalar_t *a = tensor1.data<scalar_t>();
      const scalar_t *x = tensor2.data<scalar_t>();
      scalar_t *y = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      HICE_CUBLAS_CHECK(cublasDgemv(handle, trans, dim1_tensor1, dim0_tensor1, 
                  &alpha, a, lda, x, incx, &beta, y, incy));
      break;
    }
    default:
      HICE_LOG(ERROR) << "This function doesn't handle types other than " 
                         "float, double, complex<float>, complex<double>";
  }
}

// This function performs mv between the first matrix slice in tensor1
// and the first matrix slice in tensor2.
// column-major is applied in cublas, we use cublasgemm(b, a, c)
// to implement gemm(a, b, c) on GPU without transpose. 

// tensor1: a Tensor with ndim >= 2
// tensor2: a Tensor with ndim >= 2
void mm(const Tensor& tensor1, const Tensor& tensor2, Tensor& result,
        MatmulOption option_a, MatmulOption option_b) {
  ScalarType sc_type_tensor1 = tensor1.scalar_type();
  ScalarType sc_type_tensor2 = tensor2.scalar_type();
  HICE_CHECK_EQ(sc_type_tensor1, sc_type_tensor2) 
      << "Both scalar types of arguments to mm must be equal";
  cublasOperation_t transa = 
      option_a == kNoTrans 
          ? CUBLAS_OP_N 
          : option_a == kTrans ? CUBLAS_OP_T : CUBLAS_OP_C;
  cublasOperation_t transb = 
      option_b == kNoTrans 
          ? CUBLAS_OP_N 
          : option_a == kTrans ? CUBLAS_OP_T : CUBLAS_OP_C;
  CUDAContext cuda_ctx(tensor1.device());
  cublasHandle_t handle = cuda_ctx.cublas_handle();
  HICE_CUBLAS_CHECK(cublasSetPointerMode(
    handle, CUBLAS_POINTER_MODE_HOST));
  const int dim0_tensor1 = tensor1.dim(-2);
  const int dim1_tensor1 = tensor1.dim(-1);
  const int dim0_tensor2 = tensor2.dim(-2);
  const int dim1_tensor2 = tensor2.dim(-1);
  const int dim1_result = result.dim(-1);
  const int m = transb == CUBLAS_OP_N ? dim1_tensor2 : dim0_tensor2;
  const int k1 = transb == CUBLAS_OP_N ? dim0_tensor2 : dim1_tensor2;
  const int k2 = transa == CUBLAS_OP_N ? dim1_tensor1 : dim0_tensor1;
  const int n = transa == CUBLAS_OP_N ? dim0_tensor1 : dim1_tensor1;
  HICE_CHECK_EQ(k1, k2) 
      << "Dimensions of arguments to mm must be matched";
  const int lda = dim1_tensor2;
  const int ldb = dim1_tensor1;
  const int ldc = dim1_result;
  const int k = k1;
  ScalarType sc_type = sc_type_tensor1;
  switch (sc_type) {
    case ScalarType::Float:{
      using scalar_t = float;
      const scalar_t *a = tensor2.data<scalar_t>();
      const scalar_t *b = tensor1.data<scalar_t>();
      scalar_t *c = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      HICE_CUBLAS_CHECK(cublasSgemm(handle, transb, transa, m, n, k,
                            &alpha, a, lda, b, ldb, &beta, c, ldc));
      break;
    }
    case ScalarType::Double:{
      using scalar_t = double;
      const scalar_t *a = tensor2.data<scalar_t>();
      const scalar_t *b = tensor1.data<scalar_t>();
      scalar_t *c = result.mutable_data<scalar_t>();
      scalar_t alpha = 1, beta = 0;
      HICE_CUBLAS_CHECK(cublasDgemm(handle, transb, transa, m, n, k,
                            &alpha, a, lda, b, ldb, &beta, c, ldc));
      break;
    }
    default:
      HICE_LOG(ERROR) << "This function doesn't handle types other than " 
                         "float, double, complex<float>, complex<double>";
  }
}

// batch matmul between tensor (ndim >= 2) and tensor (dim >= 3) 
void bmm(const Tensor& tensor1, const Tensor& tensor2, Tensor& result,
         MatmulOption option_a, MatmulOption option_b) {
  // std::cout << "In bmm" << std::endl; 
  ConstIntArrayRef dims_batch_tensor1(tensor1.dims().data(), std::max<int64_t>(tensor1.ndim() - 2, 0));
  ConstIntArrayRef dims_batch_tensor2(tensor2.dims().data(), std::max<int64_t>(tensor2.ndim() - 2, 0));
  ConstIntArrayRef dims_batch_result(result.dims().data(), std::max<int64_t>(result.ndim() - 2, 0));
  size_t ndim_batch_result = dims_batch_result.size();

  auto strides_from_dims = [](ConstIntArrayRef dims) -> std::vector<int64_t> {
    std::vector<int64_t> strides(dims.size(), 0);
    int64_t stride = 1;
    for (int i = dims.size() - 1; i >= 0 ; --i) {
      strides[i] = stride;
      stride *= dims[i];
    }
    return strides;
  };
  std::vector<int64_t> strides_batch_tensor1 = strides_from_dims(dims_batch_tensor1);
  std::vector<int64_t> strides_batch_tensor2 = strides_from_dims(dims_batch_tensor2);
  std::vector<int64_t> strides_batch_result = strides_from_dims(dims_batch_result);
  std::vector<int64_t> strides_for_cpt_bt1 = 
            ExpressionUtil::strides_for_computing(strides_batch_tensor1,
                                              dims_batch_tensor1,
                                              ndim_batch_result);
  std::vector<int64_t> strides_for_cpt_bt2 = 
            ExpressionUtil::strides_for_computing(strides_batch_tensor2,
                                              dims_batch_tensor2,
                                              ndim_batch_result);
  std::vector<int64_t> strides_for_cpt_bres = 
            ExpressionUtil::strides_for_computing(strides_batch_result,
                                              dims_batch_result,
                                              ndim_batch_result);

  IndexHelper idx_hlpr_tensor1(dims_batch_result, strides_for_cpt_bt1);
  IndexHelper idx_hlpr_tensor2(dims_batch_result, strides_for_cpt_bt2);
  IndexHelper idx_hlpr_result(dims_batch_result, strides_for_cpt_bres);

  Shape shape_mat1 = ShapeUtil::make_shape({tensor1.dim(-2), tensor1.dim(-1)});
  Shape shape_mat2 = ShapeUtil::make_shape({tensor2.dim(-2), tensor2.dim(-1)});
  Shape shape_mat3 = ShapeUtil::make_shape({result.dim(-2), result.dim(-1)});
  int64_t size_mat1 = ShapeUtil::get_num_items(shape_mat1);
  int64_t size_mat2 = ShapeUtil::get_num_items(shape_mat2);
  int64_t size_mat3 = ShapeUtil::get_num_items(shape_mat3);
  size_t size_batches = hice::size_from_dim(0, dims_batch_result);
  for (size_t i = 0; i < size_batches; ++i) {
    // std::cout << "#batch = " << i << std::endl; 
    auto idx_mat1 = idx_hlpr_tensor1.linear_index_to_offset(i);
    auto idx_mat2 = idx_hlpr_tensor2.linear_index_to_offset(i);
    auto idx_mat_res = idx_hlpr_result.linear_index_to_offset(i);
    Tensor mat_tensor1 = make_tensor<TensorImpl>(
        shape_mat1, tensor1.storage(), tensor1.offset() + size_mat1 * idx_mat1);
    Tensor mat_tensor2 = make_tensor<TensorImpl>(
        shape_mat2, tensor2.storage(), tensor2.offset() + size_mat2 * idx_mat2);
    Tensor mat_result = 
        make_tensor<TensorImpl>(shape_mat3, result.storage(),
                                result.offset() + size_mat3 * idx_mat_res);
    mm(mat_tensor1, mat_tensor2, mat_result, option_a, option_b);
  }
}

void matmul_impl(const Tensor &tensor1_, const Tensor &tensor2_, Tensor &result,
                 MatmulOption option_a, MatmulOption option_b, bool resizable) {
  // std::cout << "Kernel: matmul_cuda_dense_dense" << std::endl;
  Tensor tensor1 = contiguous(tensor1_);
  Tensor tensor2 = contiguous(tensor2_);
  auto ndim_tensor1 = tensor1.ndim();
  auto ndim_tensor2 = tensor2.ndim();
  if (ndim_tensor1 == 1 && ndim_tensor2 == 1) {
    std::vector<int64_t> dims_result = {1};
    ExpressionUtil::may_resize_result(result, dims_result, resizable);
    vv(tensor1, tensor2, result);
  } else if (ndim_tensor1 == 2 && ndim_tensor2 == 1) {
    int64_t dim_m =
        option_a == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
    std::vector<int64_t> dims_result = {dim_m};
    ExpressionUtil::may_resize_result(result, dims_result, resizable);
    mv(tensor1, tensor2, result, option_a);
  } else if (ndim_tensor1 == 2 && ndim_tensor2 == 2) {
    int64_t dim_m =
        option_a == kNoTrans ? tensor1.dim(0) : tensor1.dim(1);
    int64_t dim_n =
        option_b == kNoTrans ? tensor2.dim(1) : tensor2.dim(0);
    std::vector<int64_t> dims_result = {dim_m, dim_n};
    ExpressionUtil::may_resize_result(result, dims_result, resizable);
    mm(tensor1, tensor2, result, option_a, option_b);
  } else if (ndim_tensor1 == 1 && ndim_tensor2 == 2) {
    int64_t dim_n =
        option_b == kNoTrans ? tensor2.dim(1) : tensor2.dim(0);
    std::vector<int64_t> dims_result = {dim_n};
    ExpressionUtil::may_resize_result(result, dims_result, resizable);
    Tensor tensor1_new = expand_dims(tensor1, 0);
    Tensor result_new = expand_dims(result, 0);
    mm(tensor1_new, tensor2, result_new, kNoTrans, option_b);
  } else if (ndim_tensor1 >= 3  && (ndim_tensor2 == 1 || ndim_tensor2 == 2)) {
    // use mm instead of bmm
    Tensor tensor1_trans = option_a == kNoTrans
                               ? tensor1
                               : transpose_matrix(tensor1);
    Tensor tensor2_trans = option_b == kNoTrans
                               ? tensor2
                               : transpose_matrix(tensor2);
    contiguous_(tensor1_trans);
    contiguous_(tensor2_trans);
    std::vector<int64_t> dims_result;
    auto dims_tensor1 = tensor1_trans.dims();
    auto dims_tensor2 = tensor2_trans.dims();
    dims_result.insert(dims_result.end(), dims_tensor1.begin(), dims_tensor1.end() - 1);
    dims_result.insert(dims_result.end(), dims_tensor2.begin() + 1, dims_tensor2.end());
    ExpressionUtil::may_resize_result(result, dims_result, resizable);

    auto last_dim_tensor1 = tensor1_trans.dim(-1);
    std::vector<int64_t> dims_tensor1_new = {tensor1_trans.size() / last_dim_tensor1, last_dim_tensor1};
    Tensor tensor1_new = reshape(tensor1_trans, dims_tensor1_new);
    Tensor tensor2_new = ndim_tensor2 == 1 ? expand_dims(tensor2_trans, -1) : tensor2_trans;
    Tensor result_new = ndim_tensor2 == 1 ? expand_dims(result, -1) : result;
    mm(tensor1_new, tensor2_new, result_new, kNoTrans,
      kNoTrans);
  } else if ((ndim_tensor1 >= 1 && ndim_tensor2 >= 1) && (ndim_tensor1 >=3 || ndim_tensor2 >=3)) {
    // use bmm
    // broadcasting, suppose that:
    //   tensor1 {[b1], m, k1},
    //   tensor2 {[b2], k2, n},
    //   [b1] is the batch dims of tensor1,
    //   [b2] is the batch dims of tensor2,
    //   broadcast([b1], [b2]) is the batch dims of result.
    ConstIntArrayRef dims_batch_tensor1(tensor1.dims().data(), std::max<int64_t>(ndim_tensor1 - 2, 0));
    ConstIntArrayRef dims_batch_tensor2(tensor2.dims().data(), std::max<int64_t>(ndim_tensor2 - 2, 0));
    std::vector<int64_t> dims_result = hice::broadcast(dims_batch_tensor1, dims_batch_tensor2);
    int64_t dim_m = option_a == kNoTrans && ndim_tensor1 > 1
                        ? tensor1.dim(-2)
                        : tensor1.dim(-1);
    int64_t dim_n =
        option_b == kNoTrans ? tensor2.dim(-1) : tensor2.dim(-2);
    if (ndim_tensor1 == 1) {
      dims_result.insert(dims_result.end(), {dim_n});
      ExpressionUtil::may_resize_result(result, dims_result, resizable);
      Tensor tensor1_new = expand_dims(tensor1, 0);
      Tensor result_new = expand_dims(result, -2);
      bmm(tensor1_new, tensor2, result_new, kNoTrans, option_b);
    } else {
      dims_result.insert(dims_result.end(), {dim_m, dim_n});
      ExpressionUtil::may_resize_result(result, dims_result, resizable);
      bmm(tensor1, tensor2, result, option_a, option_b);
    }
  } else {
    HICE_LOG(ERROR) << "Both arguments to matmul need to be at least 1D, but they are "
                    << ndim_tensor1 <<  "D and " << ndim_tensor2 << "D";

  }
}

} // anonymous namespace

HICE_REGISTER_KERNEL(matmul_dispatcher, &matmul_impl,
                     {kCUDA, kDense},  // first operand
                     {kCUDA, kDense},  // second operand
                     {kCUDA, kDense}   // result
);

} // namespace hice
