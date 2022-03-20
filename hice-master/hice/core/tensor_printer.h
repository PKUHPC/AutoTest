// This file is based on caffe2\core\tensor.h from PyTorch.
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage.

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "hice/core/tensor.h"
#include "hice/core/sparse_tensor.h"

namespace hice {

constexpr int k_limit_default = 180;

class HICE_API TensorPrinter {
 public:
  explicit TensorPrinter(
      const std::string& tensor_name = "",
      const std::string& file_name = "",
      int limit = k_limit_default);

  ~TensorPrinter();

  void print(const Tensor& tensor);

  template <class T>
  void print(const Tensor& tensor);

  template <class T>
  void print_matrix(const Tensor& tensor,
                    int64_t n_row_limited = 20,
                    int64_t n_col_limited = 20);

  void print_meta(const Tensor& tensor);

  std::string meta_string(const Tensor& tensor);

 private:
  bool to_file_;
  int limit_;
  std::unique_ptr<std::ofstream> log_file_;
  std::string tensor_name_;
};

template <class T>
void TensorPrinter::print(const Tensor& tensor) {
  std::stringstream values_stream;
  // One most likely doesn't want to print int64_t-number of items for visual
  // inspection, so we cast down to int here.
  int total_count = static_cast<int>(std::min(tensor.size(), int64_t(limit_)));

  // Copy data to CPU for printing if it's on device
  // HICE_CHECK_EQ(tensor.device().type(), DeviceType::CPU);
  Tensor h_tensor = tensor.device().is_cpu() ? tensor : tensor.to(kCPU);

  if (h_tensor.scalar_type() == kBool) values_stream << std::boolalpha;

  // for dense
  if (h_tensor.layout_type() == kDense) {
    const T* data = h_tensor.data<T>();
    values_stream << "\n\t" << "data: ";
    for (int i = 0; i < total_count; ++i) {
      values_stream << data[i] << ", ";
    }
  }
  // for sparse
#if 1
  if (h_tensor.layout_type() == kCOO) {

    int64_t nnz = h_tensor.n_nonzero();
    total_count = static_cast<int>(std::min(nnz, int64_t(limit_)));
    const int* indices = h_tensor.indices().data<int>();

    values_stream << "\n\t" << "indices: ";
    for (int i = 0; i < total_count * tensor.ndim(); ++i) {
      values_stream << indices[i] << ", ";
    }

    const T* val_data = h_tensor.values().data<T>();
    values_stream << "\n\t" << "     values: ";
    for (int i = 0; i < total_count; ++i) {
      values_stream << val_data[i] << ", ";
    }
  }
#endif
#if 1
  if (h_tensor.layout_type() == kCSR) {
    int64_t nnz = h_tensor.n_nonzero();
    int64_t n_rows = h_tensor.dim(0);
    int total_count_for_row = static_cast<int>(std::min(n_rows + 1, int64_t(limit_)));
    total_count = static_cast<int>(std::min(nnz, int64_t(limit_)));
    const int* column_indices = h_tensor.column_indices().data<int>();
    const int* row_offsets = h_tensor.row_offsets().data<int>();
    values_stream << "\n\t" << "row_offsets: ";
    if(total_count != 0) {
      for (int i = 0; i < total_count_for_row; ++i) {
        values_stream << row_offsets[i] << ", ";
      }
    }
    values_stream << "\n\t" << "column_indices: ";
    for (int i = 0; i < total_count; ++i) {
      values_stream << column_indices[i] << ", ";
    }
    const T* val_data = h_tensor.values().data<T>();
    values_stream << "\n\t" << "values: ";
    for (int i = 0; i < total_count; ++i) {
      values_stream << val_data[i] << ", ";
    }
  }
#endif

  if (to_file_) {
    (*log_file_) << meta_string(tensor) << values_stream.str() << std::endl;
  } else {
    // Log to console.
    //HICE_LOG(INFO) << meta_string(tensor) << values_stream.str();
    std::cout << meta_string(tensor) << values_stream.str() << std::endl;
  }
}

// n_row_limited and n_col_limited controls the output range,
// if the shape is greater than (n_row_limited, n_col_limited),
// then sub matrix tensor[0:n_row_limited, 0:n_col_limited] will be printed.
template <class T>
void TensorPrinter::print_matrix(const Tensor& tensor,
                                 int64_t n_row_limited,
                                 int64_t n_col_limited) {
  std::stringstream values_stream;
  Tensor h_tensor = tensor.device_type() == kCPU ? tensor : tensor.to(kCPU);

  if (h_tensor.scalar_type() == kBool) values_stream << std::boolalpha;

  // for dense
  if (h_tensor.layout_type() == kDense) {
    auto n_row = std::min(tensor.dim(0), n_row_limited);
    auto n_col = std::min(tensor.dim(1), n_col_limited);
    auto stride = tensor.stride(0);
    const T* data = h_tensor.data<T>();
    values_stream << "\n\t" << "data: ";
    for (int i = 0; i < n_row; ++i) {
      for (int j = 0; j < n_col; ++j) {
        values_stream << data[i * stride + j] << ",";
      }
      if (i != n_row - 1){
        values_stream << "\n\t" << "      ";
      }
    }
  }
  // for sparse
#if 1
  if (h_tensor.layout_type() == kCOO) {

    int64_t nnz = h_tensor.n_nonzero();
    int64_t total_count = static_cast<int>(std::min(nnz, int64_t(limit_)));
    const int* row_indices = h_tensor.indices().data<int>();

    values_stream << "\n\t" << "row_indices: ";
    for (int i = 0; i < total_count; ++i) {
      values_stream << row_indices[i] << ", ";
    }
    const int* col_indices = row_indices + nnz;
    values_stream << "\n\t" << "col_indices: ";
    for (int i = 0; i < total_count; ++i) {
      values_stream << col_indices[i] << ", ";
    }

    const T* val_data = h_tensor.values().data<T>();
    values_stream << "\n\t" << "     values: ";
    for (int i = 0; i < total_count; ++i) {
      values_stream << val_data[i] << ", ";
    }
  }
#endif

  if (to_file_) {
    (*log_file_) << meta_string(tensor) << values_stream.str() << std::endl;
  } else {
    // Log to console.
    //HICE_LOG(INFO) << meta_string(tensor) << values_stream.str();
    std::cout << meta_string(tensor) << values_stream.str() << std::endl;
  }
}

} // namespace hice
