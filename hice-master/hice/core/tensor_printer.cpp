#include "hice/core/dispatch.h"
#include "hice/core/tensor_printer.h"

namespace hice {

TensorPrinter::TensorPrinter(
    const std::string& tensor_name,
    const std::string& file_name,
    int limit)
    : to_file_(!file_name.empty()),
      limit_(limit ? limit : k_limit_default),
      tensor_name_(tensor_name) {
  if (to_file_) {
    // We will output to file instead of printing on screen.
    // We will write each individual tensor to its individual file.
    log_file_.reset(new std::ofstream(
        file_name, std::ofstream::out | std::ofstream::trunc));
    HICE_CHECK(log_file_->good()) << "Failed to open TensorPrinter file "
        << file_name << ". rdstate() = " << log_file_->rdstate();
  }
}

TensorPrinter::~TensorPrinter() {
  if (log_file_.get()) {
    log_file_->close();
  }
}

void TensorPrinter::print(const Tensor& tensor) {
  HICE_DISPATCH_ALL_AND_COMPLEX_TYPES(tensor.scalar_type(), "tensor printer",
      [&] {
        if (tensor.is_dense() && tensor.ndim() == 2) {
          print_matrix<scalar_t>(tensor);
        } else {
          print<scalar_t>(tensor);
        }
      }
  );
}

void TensorPrinter::print_meta(const Tensor& tensor) {
  if (to_file_) {
    (*log_file_) << meta_string(tensor) << std::endl;
  } else {
    HICE_LOG(INFO) << meta_string(tensor);
  }
}

std::string TensorPrinter::meta_string(const Tensor& tensor) {
  std::stringstream meta_stream;
  meta_stream << "Tensor " << tensor_name_ << " of type "
              << tensor.data_type().name() << " of storage "
              << tensor.layout_type() << " on "
              << tensor.device() << ":\n\tdims: ";
  ConstIntArrayRef dims = tensor.dims();
  for (int i = 0; i < dims.size(); ++i) {
    meta_stream << dims[i] << ",";
  }
  // for (int i = 0; i < dims.size() - 1; ++i) {
  //   meta_stream << dims[i] << ",";
  // }
  // meta_stream << dims[dims.size() - 1];
  return meta_stream.str();
}


} // namespace hice
