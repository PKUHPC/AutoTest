#include "hice/basic/factories.h"
#include "hice/device/cuda/context_cuda.h"

namespace hice {

namespace {

template <typename T>
__global__ void uniform_shift(T *x, const int64_t n, const T min, const T max) {
  T scale = max - min;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    x[i] = x[i] * scale + min;
  }
}

void rand_uniform_dense(Tensor &tensor, Scalar min, Scalar max) {
  // std::cout << "In CUDA rand real dense" << std::endl;
  CUDAContext cuda_ctx(tensor.device());
  auto rand_generator = cuda_ctx.rand_generator();
  auto scalar_type = tensor.scalar_type();
  auto size = tensor.size();
  switch (scalar_type) {
    case ScalarType::Float: {
      auto data_ptr = tensor.mutable_data<float>();
      HICE_CURAND_CHECK(curandGenerateUniform(rand_generator, data_ptr, size));
      int num_threads = cuda_get_1d_num_threads();
      int num_blocks = cuda_get_1d_num_blocks(size);
      uniform_shift<float><<<num_blocks, num_threads, 0, cuda_ctx.stream()>>>(
          data_ptr, size, min.to<float>(), max.to<float>());
      break;
    }
    case ScalarType::Double: {
      auto data_ptr = tensor.mutable_data<double>();
      HICE_CURAND_CHECK(
          curandGenerateUniformDouble(rand_generator, data_ptr, size));
      int num_threads = cuda_get_1d_num_threads();
      int num_blocks = cuda_get_1d_num_blocks(size);
      uniform_shift<double><<<num_blocks, num_threads, 0, cuda_ctx.stream()>>>(
          data_ptr, size, min.to<double>(), max.to<double>());
      break;
    }
    default:
      HICE_LOG(ERROR) << "rand_uniform is not implemented for '"
                      << to_string(scalar_type) << "'";
  };
}

void rand_normal_dense(Tensor &tensor, Scalar mean, Scalar stddev) {
  // std::cout << "In CUDA rand real dense" << std::endl;
  CUDAContext cuda_ctx(tensor.device());
  auto rand_generator = cuda_ctx.rand_generator();
  auto scalar_type = tensor.scalar_type();
  auto size = tensor.size();
  switch (scalar_type) {
    case ScalarType::Float: {
      using scalar_t = float;
      auto data_ptr = tensor.mutable_data<scalar_t>();
      HICE_CURAND_CHECK(curandGenerateNormal(rand_generator, data_ptr, size,
                                             mean.to<scalar_t>(),
                                             stddev.to<scalar_t>()));
      break;
    }
    case ScalarType::Double: {
      using scalar_t = double;
      auto data_ptr = tensor.mutable_data<scalar_t>();
      HICE_CURAND_CHECK(curandGenerateNormalDouble(rand_generator, data_ptr,
                                                   size, mean.to<scalar_t>(),
                                                   stddev.to<scalar_t>()));
      break;
    }
    default:
      HICE_CHECK_INTERNAL(false) << "Unrecognized type.";
  };
}

}  // anonymous namespace

HICE_REGISTER_KERNEL(rand_uniform_kernel_dispatcher, &rand_uniform_dense,
                     {kCUDA, kDense});

HICE_REGISTER_KERNEL(rand_normal_kernel_dispatcher, &rand_normal_dense,
                     {kCUDA, kDense});
}  // namespace hice
