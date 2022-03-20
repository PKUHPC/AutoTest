#include "hice/ml/knn.h"

namespace hice {

HICE_DEFINE_DISPATCHER(knn_dispatcher);

Tensor knn(const Tensor& ref, const Tensor& labels, const Tensor& query, int k) {
  Tensor result(device(labels.device()).dtype(labels.data_type()));
  knn_dispatcher(ref, labels, query, k, result);
  return result;
}

} // namespce hice
