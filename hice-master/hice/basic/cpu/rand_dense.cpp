#include "hice/basic/factories.h"
#include "hice/device/cpu/context_cpu.h"

namespace hice {

namespace {

template <typename scalar_t,
          typename ext::enable_if_t<std::is_integral<scalar_t>::value, int> = 0>
std::uniform_int_distribution<scalar_t> uniform_distribution(Scalar min,
                                                             Scalar max) {
  return std::uniform_int_distribution<scalar_t>(min.to<scalar_t>(),
                                                 max.to<scalar_t>() - 1);
}

template <
    typename scalar_t,
    typename ext::enable_if_t<std::is_floating_point<scalar_t>::value, int> = 0>
std::uniform_real_distribution<scalar_t> uniform_distribution(Scalar min,
                                                              Scalar max) {
  return std::uniform_real_distribution<scalar_t>(min.to<scalar_t>(),
                                                  max.to<scalar_t>());
}

void rand_uniform_dense(Tensor &tensor, Scalar min, Scalar max) {
  // std::cout << "In CPU rand dense" << std::endl;
  ScalarType scalar_type = tensor.scalar_type();
  HICE_DISPATCH_ALL_TYPES(scalar_type, "cpu_rand_uniform_dense", [&] {
    CPUContext cpu_ctx;
    auto rand_generator = cpu_ctx.rand_generator();
    auto size = tensor.size();
    auto data_ptr = tensor.mutable_data<scalar_t>();
    auto distribution = uniform_distribution<scalar_t>(min, max);
    for (int64_t i = 0; i < size; ++i) {
      data_ptr[i] = distribution(rand_generator);
    }
  });
}

void rand_normal_dense(Tensor &tensor, Scalar mean, Scalar stddev) {
  // std::cout << "In CPU rand dense" << std::endl;
  ScalarType scalar_type = tensor.scalar_type();
  switch (scalar_type) {
    case kFloat: {
      using scalar_t = float;
      CPUContext cpu_ctx;
      auto rand_generator = cpu_ctx.rand_generator();
      auto size = tensor.size();
      auto data_ptr = tensor.mutable_data<scalar_t>();
      auto distribution = std::normal_distribution<scalar_t>(
          mean.to<scalar_t>(), stddev.to<scalar_t>());
      for (int64_t i = 0; i < size; ++i) {
        data_ptr[i] = distribution(rand_generator);
      }
      break;
    }
    case kDouble: {
      using scalar_t = double;
      CPUContext cpu_ctx;
      auto rand_generator = cpu_ctx.rand_generator();
      auto size = tensor.size();
      auto data_ptr = tensor.mutable_data<scalar_t>();
      auto distribution = std::normal_distribution<scalar_t>(
          mean.to<scalar_t>(), stddev.to<scalar_t>());
      for (int64_t i = 0; i < size; ++i) {
        data_ptr[i] = distribution(rand_generator);
      }
      break;
    }
    default:
      HICE_CHECK_INTERNAL(false) << "Unrecognized type.";
  }
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(rand_uniform_kernel_dispatcher, &rand_uniform_dense,
                     {kCPU, kDense});

HICE_REGISTER_KERNEL(rand_normal_kernel_dispatcher, &rand_normal_dense,
                     {kCPU, kDense});

}  // namespace hice