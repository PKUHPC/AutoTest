#pragma once
namespace hice {

template <typename TScalarType, typename TIndexType>
__global__ void compare_kernel(const TScalarType* data_in,
                               TScalarType* data_out, TIndexType* data_indices,
                               int64_t stride, int batch, int n, bool greater) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= batch * stride){ 
    return;
  }
  int64_t b = idx / stride;
  int64_t i = idx % stride;
  const TScalarType* data = &data_in[b * n * stride + i];
  TScalarType result = data[0];
  TIndexType result_index = 0; 
  for(int i = 0;i < n;i++){
    TScalarType value = data[i * stride];
    bool cmp = greater ? (result > value) : (result < value);
    result = cmp ? result : value;
    result_index = cmp ? result_index : i;
  }
  data_out[b * stride + i] = result;
  data_indices[b * stride + i] = result_index;
}

void test(){
  printf("In compare kernel\n");
}
template <typename scalar_t, typename index_t>
static void launch_compare(Tensor& res, Tensor& res_indices, const Tensor& self,
                    int64_t reduce_dim, bool greater) {
  auto data_out = res.mutable_data<scalar_t>();
  auto data_indices = res_indices.mutable_data<index_t>();
  auto data_in = self.data<scalar_t>();
  auto numel = self.size();

  int64_t n = self.dim(reduce_dim);
  int64_t stride = self.stride(reduce_dim);

  if (n == 1) {
    stride = 1;
    for (int64_t i = self.ndim() - 1; i > reduce_dim; i--) {
      stride *= self.dim(i);
    }
  }
  int64_t batch = numel / (n * stride);
  const int block_size = 64;
  const int num_blocks = n / block_size + 1;

  compare_kernel<<<num_blocks, block_size>>>(data_in, data_out, data_indices,
                                             stride, batch, n, greater);
}
}  // namespace hice