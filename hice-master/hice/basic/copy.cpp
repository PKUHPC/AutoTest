#include "hice/basic/copy.h"

namespace hice {

HICE_DEFINE_DISPATCHER(copy_dispatcher);

Tensor& copy(const Tensor &src, Tensor &dst, bool non_blocking) {
  HICE_CHECK(src.is_dense() && dst.is_dense()) 
    << "copy only supports dense tensor, if you want to copy sparse, use clone() instead";
  if (!src.is_same(dst)) {
    copy_dispatcher(src, dst, non_blocking);
  }
  return dst;
}

} // namespce hice