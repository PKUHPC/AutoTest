#pragma once

#include "hice/math/cuda/atomics.cuh"
#include "hice/math/cuda/offset_calculator.cuh"

namespace hice {

// returns floor(log2(n))
static inline int last_pow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

static inline int64_t div_up(int64_t a, int64_t b) { return (a + b - 1) / b; }

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta,
                                            int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

struct ReduceConfig {
  ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
      : element_size_bytes(element_size_bytes),
        num_inputs(num_inputs),
        num_outputs(num_outputs) {}

  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;
  static constexpr int MAX_THREAD_IN_BLOCK = 512;

  void set_block_dimension(int64_t dim0, int64_t dim1) {
    int dim0_pow2 = dim0 < MAX_THREAD_IN_BLOCK
                        ? static_cast<int>(last_pow2(dim0))
                        : MAX_THREAD_IN_BLOCK;
    int dim1_pow2 = dim1 < MAX_THREAD_IN_BLOCK
                        ? static_cast<int>(last_pow2(dim1))
                        : MAX_THREAD_IN_BLOCK;
    block_width = std::min(dim0_pow2, warp_size);
    block_height = std::min(dim1_pow2, int(MAX_THREAD_IN_BLOCK / block_width));
    block_width = std::min(dim0_pow2, int(MAX_THREAD_IN_BLOCK / block_height));
    num_threads_in_block = block_width * block_height;
  }

  int split_input(int new_step) {
    int step = step_input;
    step_input *= new_step;
    return step;
  }

  int split_output(int new_step) {
    int step = step_output;
    step_output *= new_step;
    return step;
  }

  int values_per_thread() const { return div_up(num_inputs, step_input); }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() || block_width <= warp_size)) {
      return 0;
    }
    return num_threads_in_block * element_size_bytes;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return sizeof(int) * grid().x;
  }

  int64_t global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
    if (!should_block_x_reduce()) {
      size *= block().x;
    }
    return size;
  }

  dim3 block() const { return dim3(block_width, block_height); }

  dim3 grid() const {
    return dim3(div_up(num_outputs, step_output), ctas_per_output);
  }

  __device__ __host__ bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  __device__ __host__ bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

  __device__ __host__ bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  __device__ int idx_in() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] + warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  __device__ int idx_out() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] + warp * output_mult[BLOCK_Y] +
            cta1 * step_output);
  }

  __device__ int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  __device__ bool should_store(int idx_out) const {
    return idx_out < num_outputs &&
           (!should_block_x_reduce() || threadIdx.x == 0) &&
           (!should_block_y_reduce() || threadIdx.y == 0);
  }

  __device__ int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  int num_inputs;
  int num_outputs;
  int block_width;
  int block_height;
  int num_threads_in_block;
  int element_size_bytes;

  int warp_size = 32;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};
};

template <typename TScalarType, typename TOp>
struct ReduceOp {
  ReduceOp(TOp op, ReduceConfig config, OffsetCalculator calc_in,
           OffsetCalculator calc_in_base, OffsetCalculator calc_out,
           const TScalarType *in, TScalarType *out, void *cta_buf,
           int *semaphores, TScalarType init_value, TScalarType factor)
      : op(op),
        config(config),
        calc_in(calc_in),
        calc_in_base(calc_in_base),
        calc_out(calc_out),
        in(in),
        out(out),
        cta_buf(cta_buf),
        semaphores(semaphores),
        init_value(init_value),
        factor(factor) {}

  __device__ TScalarType block_y_reduce(TScalarType value,
                                        char *shared_memory) const {
    TScalarType *shared = (TScalarType *)shared_memory;
    shared[config.shared_memory_offset(0)] = value;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
        TScalarType other = shared[config.shared_memory_offset(offset)];
        value = op(value, other);
        shared[config.shared_memory_offset(0)] = value;
      }
    }
    return value;
  }

  __device__ TScalarType block_x_reduce(TScalarType value,
                                        char *shared_memory) const {
    int dim_x = blockDim.x;
    TScalarType *shared = (TScalarType *)shared_memory;
    if (dim_x > warpSize) {
      int address_base = threadIdx.x + threadIdx.y * blockDim.x;
      shared[address_base] = value;
      for (int offset = dim_x / 2; offset >= warpSize; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
          TScalarType other = shared[address_base + offset];
          value = op(value, other);
          shared[address_base] = value;
        }
      }
      dim_x = warpSize;
    }

    __syncthreads();

    for (int offset = 1; offset < dim_x; offset <<= 1) {
      TScalarType other = WARP_SHFL_DOWN(value, offset);
      value = op(value, other);
    }
    return value;
  }

  __device__ TScalarType load_input(const TScalarType *in, int offset) const
  {
    int stride = calc_in.strides_[0];
    if (calc_in.ndim == 1) {
      return in[offset * stride];
    } else {
      return in[calc_in.idx2offset(offset)];
    }
  }

  __device__ TScalarType thread_reduce(const TScalarType *data) const {
    TScalarType value = init_value;
    int idx = config.idx_in();
    while (idx < config.num_inputs) {
      TScalarType next_value = load_input(data, idx);
      value = op(value, next_value);
      idx += config.step_input;
    }
    return value;
  }

  __device__ TScalarType global_reduce(TScalarType value, TScalarType *out,
                                       char *shared_memory) const {
    TScalarType *reduce_buffer = (TScalarType *)cta_buf;
    bool should_store = config.should_store(config.idx_out());
    if (should_store) {
      int offset = config.staging_memory_offset(blockIdx.y);
      reduce_buffer[offset] = value;
    }
    __threadfence();
    __syncthreads();

    bool is_last_block_done = mark_block_finished();
    if (is_last_block_done) {
      value = init_value;
      if (config.should_block_x_reduce()) {
        int input_offset = threadIdx.x + threadIdx.y * blockDim.x;
        int step = blockDim.x * blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          int idx = config.staging_memory_offset(input_offset);
          TScalarType next = reduce_buffer[idx];
          value = op(value, next);
        }
      } else {
        int input_offset = threadIdx.y;
        int step = blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          int idx = config.staging_memory_offset(input_offset);
          TScalarType next = reduce_buffer[idx];
          value = op(value, next);
        }
      }
      value = block_y_reduce(value, shared_memory);
      if (config.should_block_x_reduce()) {
        value = block_x_reduce(value, shared_memory);
      }
      // if (should_store) {
      //   *out = value;
      // }
      *out = should_store ? value : *out;
    }
    return value;
  }

  __device__ void run() const {
    extern __shared__ char shared_memory[];

    int idx_out = config.idx_out();
    int idx_in = config.idx_in();

    int base_offset = calc_out.idx2offset(idx_out);
    TScalarType value = init_value;
    if (idx_out < config.num_outputs && idx_in < config.num_inputs) {
      auto slice_in = in + calc_in_base.idx2offset(idx_out);
      value = thread_reduce(slice_in);
    }

    if (config.should_block_y_reduce()) {
      value = block_y_reduce(value, shared_memory);
    }
    if (config.should_block_x_reduce()) {
      value = block_x_reduce(value, shared_memory);
    }

    auto ptr_out = out + base_offset;
    if (config.should_global_reduce()) {
      value = global_reduce(value, ptr_out, shared_memory);
      *ptr_out = value / factor;
    } else if (config.should_store(idx_out)) {
      *ptr_out = value / factor;
    }
  }

  __device__ bool mark_block_finished() const {
    __shared__ bool is_last_block_done_shared;
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
    }
    __syncthreads();
    return is_last_block_done_shared;
  }

  TOp op;
  const TScalarType *in;
  TScalarType *out;
  ReduceConfig config;
  OffsetCalculator calc_in;
  OffsetCalculator calc_in_base;
  OffsetCalculator calc_out;
  void *cta_buf;
  int *semaphores;
  TScalarType init_value, factor;

};  // end of ReduceOp

template <typename TReduction>
__global__ void reduce_kernel(TReduction reduction) {
  reduction.run();
}

// template <typename TScalarType, typename TOp>
// __global__ void reduce_kernel(const TScalarType *in, TScalarType *out,
//                               TScalarType init_value, int factor,
//                               int num_outputs, int inputs_per_output,
//                               OffsetCalculator calc_in,
//                               OffsetCalculator calc_in_base,
//                               OffsetCalculator calc_out, TOp op) {
//   int idx_out = blockDim.x * blockIdx.x + threadIdx.x;
//   if(idx_out >= num_outputs){
//     return;
//   }
//   int base_offset = calc_out.idx2offset(idx_out);
//   auto slice_in = in + calc_in_base.idx2offset(idx_out);
//   TScalarType value = init_value;
//   for(int i = 0;i < inputs_per_output;i++){
//     TScalarType next_value = slice_in[calc_in.idx2offset(i)];
//     value = op(value, next_value);
//   }
//   out[base_offset] = value / factor;
// }
}  // namespace hice
