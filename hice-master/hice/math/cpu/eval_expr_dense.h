#pragma once

#include "hice/core/tensor.h"
#include "hice/math/cpu/openmp/parallel.h"
#include "hice/math/cpu/loop_kernel_dense.h"
#include <sys/time.h>

namespace hice {

template <typename TScalarType1, typename TScalarType2, typename TOp>
void eval_unary_expr(Expression& expr, TOp op) {
  // std::cout<<"In eval_unary_expr"<<std::endl;
  const Tensor& tensor = expr.input(0);
  Tensor& result = expr.output(0);
  auto data_ptr_tensor = tensor.data<TScalarType1>();
  auto data_ptr_result = result.mutable_data<TScalarType2>();
  ConstIntArrayRef dims_tensor = tensor.dims();
  ConstIntArrayRef dims_result = result.dims();
  auto ndim_result = result.ndim();
  ConstIntArrayRef strides_tensor = expr.strides_input(0);
  ConstIntArrayRef strides_result = expr.strides_output(0);
  IndexHelper idx_hlpr_tensor(dims_tensor, strides_tensor);
  IndexHelper idx_hlpr_result(dims_result, strides_result);

  int64_t size = result.size();
  if (size < hice::GRAIN_SIZE || hice::get_max_threads() == 1) {
    serial_unary_loop_kernel(data_ptr_tensor, data_ptr_result,
                             idx_hlpr_tensor, idx_hlpr_result,
                             0, size, op);
  } else {
    parallel_for(0, size, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      serial_unary_loop_kernel(data_ptr_tensor, data_ptr_result,
                               idx_hlpr_tensor, idx_hlpr_result,
                               begin, end, op);
    });
  }
}

template <typename TScalarType1, typename TScalarType2, typename TOp, typename TVOp>
void eval_unary_expr_vec(Expression& expr, TOp op, TVOp vec_op) {
  // std::cout<<"In eval_unary_expr"<<std::endl;
  // data preparation
  const Tensor& tensor = expr.input(0);
  Tensor& result = expr.output(0);

  bool vectorizable = platform_support_vec()                       && 
                      tensor.is_default_layout()                   &&
                      result.is_default_layout()                   &&
                      tensor.scalar_type() == result.scalar_type() &&
                      tensor.size() == result.size();
  int64_t size = result.size();
  auto data_ptr_tensor = tensor.data<TScalarType1>();
  auto data_ptr_result = result.mutable_data<TScalarType2>();
  
#if 1

  if (vectorizable) {
    if (size < hice::GRAIN_SIZE || hice::get_max_threads() == 1) {
      serial_unary_loop_kernel_vec(data_ptr_tensor, data_ptr_result,
                                    0, size, op, vec_op);
    } else {
      parallel_for(0, size, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        serial_unary_loop_kernel_vec(data_ptr_tensor, data_ptr_result,
                                    begin, end, op, vec_op);
      });
    }
  } else {
    ConstIntArrayRef dims_tensor = tensor.dims();
    ConstIntArrayRef dims_result = result.dims();
    ConstIntArrayRef strides_tensor = expr.strides_input(0);
    ConstIntArrayRef strides_result = expr.strides_output(0);
    IndexHelper idx_hlpr_tensor(dims_tensor, strides_tensor);
    IndexHelper idx_hlpr_result(dims_result, strides_result);
    if (size < hice::GRAIN_SIZE || hice::get_max_threads() == 1) {
      serial_unary_loop_kernel(data_ptr_tensor, data_ptr_result,
                              idx_hlpr_tensor, idx_hlpr_result,
                              0, size, op);
    } else {
      parallel_for(0, size, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        serial_unary_loop_kernel(data_ptr_tensor, data_ptr_result,
                                idx_hlpr_tensor, idx_hlpr_result,
                                begin, end, op);
      });
    }
  }

#else
  timeval t1, t2;
  double time_cost = 0;
  const int kRUN = 50;

  ConstIntArrayRef dims_tensor = tensor.dims();
  ConstIntArrayRef dims_result = result.dims();
  ConstIntArrayRef strides_tensor = expr.strides_input(0);
  ConstIntArrayRef strides_result = expr.strides_output(0);
  IndexHelper idx_hlpr_tensor(dims_tensor, strides_tensor);
  IndexHelper idx_hlpr_result(dims_result, strides_result);


#if defined(__AVX__) && !defined(_MSC_VER)
  // std::cout<<"yes"<<std::endl;
#endif

  gettimeofday(&t1,NULL); 
  for (int i = 0; i < kRUN; ++i) {
    serial_unary_loop_kernel_basic(data_ptr_tensor, data_ptr_result,
                                  0, size, op);
  }
  gettimeofday(&t2,NULL); 
  time_cost = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / kRUN;
  std::cout << "Time cost(serial basic): " << time_cost << "s" << std::endl;

  gettimeofday(&t1,NULL); 
  for (int i = 0; i < kRUN; ++i) {
    serial_unary_loop_kernel_vec(data_ptr_tensor, data_ptr_result,
                                  0, size, op, vec_op);
  }
  gettimeofday(&t2,NULL); 
  time_cost = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / kRUN;
  std::cout << "Time cost(serial vec): " << time_cost << "s" << std::endl;

  gettimeofday(&t1,NULL); 
  for (int i = 0; i < kRUN; ++i) {
    serial_unary_loop_kernel(data_ptr_tensor, data_ptr_result,
                            idx_hlpr_tensor, idx_hlpr_result,
                            0, size, op);
  }
  gettimeofday(&t2,NULL); 
  time_cost = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / kRUN;
  std::cout << "Time cost(serial stride): " << time_cost << "s" << std::endl;

  gettimeofday(&t1,NULL); 
  for (int i = 0; i < kRUN; ++i) {
    parallel_for(0, size, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      serial_unary_loop_kernel_basic(data_ptr_tensor, data_ptr_result,
                                    begin, end, op);
    });
  }
  gettimeofday(&t2,NULL); 
  time_cost = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / kRUN;
  std::cout << "Time cost(parallel basic): " << time_cost << "s" << std::endl;

  gettimeofday(&t1,NULL); 
  for (int i = 0; i < kRUN; ++i) {
    parallel_for(0, size, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      serial_unary_loop_kernel_vec(data_ptr_tensor, data_ptr_result,
                                  begin, end, op, vec_op);
    });
  }
  gettimeofday(&t2,NULL); 
  time_cost = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / kRUN;
  std::cout << "Time cost(parallel vec): " << time_cost << "s" << std::endl;

  gettimeofday(&t1,NULL); 
  for (int i = 0; i < kRUN; ++i) {
    parallel_for(0, size, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      serial_unary_loop_kernel(data_ptr_tensor, data_ptr_result,
                              idx_hlpr_tensor, idx_hlpr_result,
                              begin, end, op);
    });
  }
  gettimeofday(&t2,NULL); 
  time_cost = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / kRUN;
  std::cout << "Time cost(parallel stride): " << time_cost << "s" << std::endl;
#endif
}

template <typename TScalarType1, typename TScalarType2, typename TScalarType3, typename TOp>
void eval_binary_expr(Expression& expr, TOp op) {
  // data preparation
  const Tensor& tensor1 = expr.input(0);
  const Tensor& tensor2 = expr.input(1);
  Tensor& result = expr.output(0);
  auto data_ptr_tensor1 = tensor1.data<TScalarType1>();
  auto data_ptr_tensor2 = tensor2.data<TScalarType2>();
  auto data_ptr_result = result.mutable_data<TScalarType3>();
  ConstIntArrayRef dims_result = result.dims();
  auto ndim_result = result.ndim();
  ConstIntArrayRef strides_tensor1 = expr.strides_input(0);
  ConstIntArrayRef strides_tensor2 = expr.strides_input(1);
  ConstIntArrayRef strides_result = expr.strides_output(0);
  IndexHelper idx_hlpr_tensor1(dims_result, strides_tensor1);
  IndexHelper idx_hlpr_tensor2(dims_result, strides_tensor2);
  IndexHelper idx_hlpr_result(dims_result, strides_result);

  int64_t size_result = result.size();
  if (size_result < hice::GRAIN_SIZE || hice::get_max_threads() == 1) {
    serial_binary_loop_kernel(data_ptr_tensor1, data_ptr_tensor2,
                              data_ptr_result, idx_hlpr_tensor1,
                              idx_hlpr_tensor2, idx_hlpr_result,
                              0, size_result, op);
  } else {
    HICE_DLOG(INFO) << "In parallel_for";
    parallel_for(0, size_result, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      serial_binary_loop_kernel(data_ptr_tensor1, data_ptr_tensor2,
                                data_ptr_result, idx_hlpr_tensor1,
                                idx_hlpr_tensor2, idx_hlpr_result,
                                begin, end, op);
    });
  }
}

template <typename TScalarType1, typename TScalarType2, typename TScalarType3, 
          typename TOp, typename TVOp>
void eval_binary_expr_vec(Expression& expr, TOp op, TVOp vec_op) {
  // data preparation
  const Tensor& tensor1 = expr.input(0);
  const Tensor& tensor2 = expr.input(1);
  Tensor& result = expr.output(0);

  bool vectorizable = platform_support_vec()            && 
                      tensor1.is_default_layout()       && 
                      tensor2.is_default_layout()       &&
                      result.is_default_layout()        &&
                      tensor1.scalar_type() == tensor2.scalar_type() &&
                      tensor1.scalar_type() == result.scalar_type() &&
                      tensor1.size() == tensor2.size()  &&
                      tensor1.size() == result.size();
  int64_t size_result = result.size();
  auto data_ptr_tensor1 = tensor1.data<TScalarType1>();
  auto data_ptr_tensor2 = tensor2.data<TScalarType2>();
  auto data_ptr_result = result.mutable_data<TScalarType3>();

#if 1

  if (vectorizable) {
    if (size_result < hice::GRAIN_SIZE || hice::get_max_threads() == 1) {
      serial_binary_loop_kernel_vec(data_ptr_tensor1, data_ptr_tensor2,
                                    data_ptr_result, 
                                    0, size_result, op, vec_op);
    } else {
      parallel_for(0, size_result, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        serial_binary_loop_kernel_vec(data_ptr_tensor1, data_ptr_tensor2,
                                      data_ptr_result, 
                                      begin, end, op, vec_op);
      });
    }
  } else {
    ConstIntArrayRef dims_result = result.dims();
    ConstIntArrayRef strides_tensor1 = expr.strides_input(0);
    ConstIntArrayRef strides_tensor2 = expr.strides_input(1);
    ConstIntArrayRef strides_result = expr.strides_output(0);
    IndexHelper idx_hlpr_tensor1(dims_result, strides_tensor1);
    IndexHelper idx_hlpr_tensor2(dims_result, strides_tensor2);
    IndexHelper idx_hlpr_result(dims_result, strides_result);
    if (size_result < hice::GRAIN_SIZE || hice::get_max_threads() == 1) {
      serial_binary_loop_kernel(data_ptr_tensor1, data_ptr_tensor2,
                                data_ptr_result, idx_hlpr_tensor1,
                                idx_hlpr_tensor2, idx_hlpr_result,
                                0, size_result, op);
    } else {
      parallel_for(0, size_result, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        serial_binary_loop_kernel(data_ptr_tensor1, data_ptr_tensor2,
                                  data_ptr_result, idx_hlpr_tensor1,
                                  idx_hlpr_tensor2, idx_hlpr_result,
                                  begin, end, op);
      });
    }
  }

#else

  ConstIntArrayRef dims_result = result.dims();
  ConstIntArrayRef strides_tensor1 = expr.strides_input(0);
  ConstIntArrayRef strides_tensor2 = expr.strides_input(1);
  ConstIntArrayRef strides_result = expr.strides_output(0);
  IndexHelper idx_hlpr_tensor1(dims_result, strides_tensor1);
  IndexHelper idx_hlpr_tensor2(dims_result, strides_tensor2);
  IndexHelper idx_hlpr_result(dims_result, strides_result);

  
  timeval t1, t2;
  double time_cost = 0;
  const int kRUN = 50;

  // gettimeofday(&t1,NULL); 
  // for (int i = 0; i < kRUN; ++i) {
  //   serial_binary_loop_kernel(data_ptr_tensor1, data_ptr_tensor2,
  //                             data_ptr_result, idx_hlpr_tensor1,
  //                             idx_hlpr_tensor2, idx_hlpr_result,
  //                             0, size_result, op);
  // }
  // gettimeofday(&t2,NULL); 
  // time_cost = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / kRUN;
  // std::cout << "Time cost(serial): " << time_cost << "s" << std::endl;

  gettimeofday(&t1,NULL); 
  for (int i = 0; i < kRUN; ++i) {
  serial_binary_loop_kernel_basic(data_ptr_tensor1, data_ptr_tensor2,
                                data_ptr_result, 
                                0, size_result, op);
  }
  gettimeofday(&t2,NULL); 
  time_cost = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / kRUN;
  std::cout << "Time cost(serial basic): " << time_cost << "s" << std::endl;

  gettimeofday(&t1,NULL); 
  for (int i = 0; i < kRUN; ++i) {
  serial_binary_loop_kernel_vec(data_ptr_tensor1, data_ptr_tensor2,
                                data_ptr_result, 
                                0, size_result, op, vec_op);
  }
  gettimeofday(&t2,NULL); 
  time_cost = (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1000000.0) * 1000 / kRUN;
  std::cout << "Time cost(serial vec): " << time_cost << "s" << std::endl;

  // start = clock();
  // serial_binary_loop_kernel_vec(data_ptr_tensor1, data_ptr_tensor2,
  //                               data_ptr_result, 
  //                               0, size_result, op, vec_op);
  // end   = clock();
  // std::cout << "Time cost(serial vec): " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

  // start = clock();
  // parallel_for(0, size_result, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
  //   serial_binary_loop_kernel(data_ptr_tensor1, data_ptr_tensor2,
  //                             data_ptr_result, idx_hlpr_tensor1,
  //                             idx_hlpr_tensor2, idx_hlpr_result,
  //                             begin, end, op);
  // });
  // end   = clock();
  // std::cout << "Time cost(parallel): " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;


  // start = clock();
  // parallel_for(0, size_result, hice::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
  //   serial_binary_loop_kernel_vec(data_ptr_tensor1, data_ptr_tensor2,
  //                                 data_ptr_result, 
  //                                 begin, end, op, vec_op);
  // });
  // end   = clock();
  // std::cout << "Time cost(parallel vec): " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

#endif
}

template <typename TScalarType, typename TOp>
void eval_reduce_expr(Expression& expr, TScalarType init_value, TOp op) {
  // data preparation
  const Tensor& self = expr.input(0);
  Tensor& result = expr.output(0);
  result.fill(init_value);
  auto data_ptr_in = self.data<TScalarType>();
  auto data_ptr_out = result.mutable_data<TScalarType>();

  int num_items = self.size();
  std::vector<int64_t> strides_in = expr.strides_input(0);
  std::vector<int64_t> strides_out = expr.strides_output(0);
  std::vector<int64_t> perm = expr.reorder_dims(strides_out);

  std::vector<int64_t> permuted_strides_in = expr.permute_dims(strides_in, perm);
  std::vector<int64_t> permuted_strides_out = expr.permute_dims(strides_out, perm);
  std::vector<int64_t> dims = expr.permute_dims(self.shape().dimensions(), perm);

  reduce_kernel(data_ptr_in, data_ptr_out, permuted_strides_in,
      permuted_strides_out, dims, num_items, op);
}

} // namespace hice