#include "hice/math/arg_reduce.h"
#include "hice/math/cpu/arg_reduce_kernel.h"

namespace hice{

namespace {

void min_impl(const Tensor &self, int64_t dim, Tensor &min, Tensor &min_indices) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "min", [&]() {
    launch_compare<scalar_t, int64_t>(min, min_indices, self, dim, false);
  });
}

void max_impl(const Tensor &self, int64_t dim, Tensor &max, Tensor &max_indices) {
  ScalarType sc_type = self.scalar_type();
  HICE_DISPATCH_ALL_TYPES(sc_type, "max", [&]() {
    launch_compare<scalar_t, int64_t>(max, max_indices, self, dim, true);
  });
}

} // anonymous namespace

HICE_REGISTER_KERNEL(min_tuple_dispatcher, &min_impl, {kCPU, kDense}, {kCPU, kDense},
                     {kCPU, kDense});

HICE_REGISTER_KERNEL(max_tuple_dispatcher, &max_impl, {kCPU, kDense}, {kCPU, kDense},
                     {kCPU, kDense});

} // namespace hice
