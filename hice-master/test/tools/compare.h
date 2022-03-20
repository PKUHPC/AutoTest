#pragma once

#include "gtest/gtest.h"
#include "hice/basic/factories.h"
#include "hice/core/sparse_tensor.h"

namespace hice {

inline void ExpectEqualDenseRegardlessDevice(const Tensor& tensor1,
                                             const Tensor& tensor2) {
  EXPECT_EQ(tensor1.size(), tensor2.size());
  EXPECT_EQ(tensor1.offset(), tensor2.offset());
  EXPECT_EQ(tensor1.data_type(), tensor2.data_type());
  EXPECT_EQ(tensor1.ndim(), tensor2.ndim());
  EXPECT_EQ(tensor1.shape(), tensor2.shape());
  EXPECT_EQ(tensor1.strides(), tensor2.strides());
  Tensor tensor1_new =
      tensor1.device_type() == kCPU ? tensor1 : tensor1.to(kCPU);
  Tensor tensor2_new =
      tensor2.device_type() == kCPU ? tensor2 : tensor2.to(kCPU);
  auto size = tensor1.size();
  auto sc_type = tensor1.scalar_type();
  for (int i = 0; i < size; ++i) {
    if (sc_type == kFloat || sc_type == kDouble) {
      HICE_DISPATCH_FLOATING_TYPES(sc_type, "", [&]() {
        EXPECT_FLOAT_EQ(tensor1_new.data<scalar_t>()[i],
                        tensor2_new.data<scalar_t>()[i]);
      });
    } else {
      HICE_DISPATCH_ALL_TYPES_AND(kBool, sc_type, "", [&]() {
        EXPECT_EQ(tensor1_new.data<scalar_t>()[i],
                  tensor2_new.data<scalar_t>()[i]);
      });
    }
  }
}

inline void ExpectEqualSparseRegardlessDevice(const Tensor& sparse_a,
                                              const Tensor& sparse_b) {
  EXPECT_EQ(sparse_a.data_type(), sparse_b.data_type());
  EXPECT_EQ(sparse_a.ndim(), sparse_b.ndim());
  EXPECT_EQ(sparse_a.shape(), sparse_b.shape());
  EXPECT_EQ(sparse_a.n_nonzero(), sparse_b.n_nonzero());
  // check values
  ExpectEqualDenseRegardlessDevice(sparse_a.values(), sparse_b.values());
  // check indices
  if (sparse_a.layout_type() == kCOO) {
    ExpectEqualDenseRegardlessDevice(sparse_a.indices(), sparse_b.indices());
  } else {
    // kCSR
    ExpectEqualDenseRegardlessDevice(sparse_a.column_indices(),
                                     sparse_b.column_indices());
    ExpectEqualDenseRegardlessDevice(sparse_a.row_offsets(),
                                     sparse_b.row_offsets());
  }
}

inline void ExpectEqualDense(const Tensor& tensor1, const Tensor& tensor2) {
  EXPECT_EQ(tensor1.device(), tensor2.device());
  ExpectEqualDenseRegardlessDevice(tensor1, tensor2);
}

inline void ExpectEqualSparse(const Tensor& sparse_a, const Tensor& sparse_b) {
  EXPECT_EQ(sparse_a.device(), sparse_b.device());
  ExpectEqualSparseRegardlessDevice(sparse_a, sparse_b);
}

#if 0
inline void ExpectEqualDenseWithError(const Tensor& tensor1, const Tensor& tensor2) {
  HICE_CHECK_EQ(tensor1.size(), tensor2.size());
  HICE_CHECK_EQ(tensor1.offset(), tensor2.offset());
  HICE_CHECK_EQ(tensor1.data_type(), tensor2.data_type());
  HICE_CHECK_EQ(tensor1.ndim(), tensor2.ndim());
  HICE_CHECK_EQ(tensor1.device(), tensor2.device());
  HICE_CHECK(tensor1.dims() == tensor2.dims());
  HICE_CHECK(tensor1.strides() == tensor2.strides());
  Tensor tensor1_new = tensor1.device_type() == kCPU ? tensor1 : tensor1.to(kCPU);
  Tensor tensor2_new = tensor2.device_type() == kCPU ? tensor2 : tensor2.to(kCPU);
  auto size = tensor1.size();
  double err_max = 0, err = 0;
  for (int i = 0; i < size; ++i) {
    err = std::abs(tensor1_new.data<double>()[i] - tensor2_new.data<double>()[i]);
    err_max = std::max(err_max, err);
  }
  bool passed = err_max < 0.1;
  if (!passed) {
    std::cout<<"max_err = "<<err_max<<std::endl;
  }
  HICE_CHECK(passed);
}
#endif

inline void ExpectEqualDenseWithError(const Tensor& tensor1,
                                      const Tensor& tensor2,
                                      Scalar threshold = (double)0.1) {
  EXPECT_EQ(tensor1.size(), tensor2.size());
  EXPECT_EQ(tensor1.offset(), tensor2.offset());
  EXPECT_EQ(tensor1.data_type(), tensor2.data_type());
  EXPECT_EQ(tensor1.ndim(), tensor2.ndim());
  EXPECT_EQ(tensor1.shape(), tensor2.shape());
  EXPECT_EQ(tensor1.strides(), tensor2.strides());
  Tensor tensor1_new = tensor1.device_type() == kCPU ? tensor1 : tensor1.to(kCPU);
  Tensor tensor2_new = tensor2.device_type() == kCPU ? tensor2 : tensor2.to(kCPU);
  auto size = tensor1.size();
  double err_max = 0, err = 0;
  auto sc_type = tensor1.scalar_type();
  for(int i = 0; i < size; ++i) {
    if (sc_type == kFloat || sc_type == kDouble) {
      HICE_DISPATCH_FLOATING_TYPES(sc_type, "", [&]() {
        err = std::abs(tensor1_new.data<scalar_t>()[i] -
                       tensor2_new.data<scalar_t>()[i]);
        err_max = std::max(err_max, err);
      });
    } else {
      HICE_DISPATCH_ALL_TYPES_AND(kBool, sc_type, "", [&]() {
        EXPECT_EQ(tensor1_new.data<scalar_t>()[i],
                  tensor2_new.data<scalar_t>()[i]);
      });
    }
  }
  bool passed = err_max < threshold.toDouble();
  if (!passed) {
    std::cout<<"max_err = "<<err_max<<std::endl;
  }
  HICE_CHECK(passed);
}

}  // namespace hice