#include "hice/ml/knn.h"
#include "hice/device/cuda/context_cuda.h"
#include "hice/core/tensor_printer.h"

namespace hice {

namespace {

#define TILE_WIDTH 32

__global__ void compute_squared_norm(const float *array, int row, int col,
                                     float *norm) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < row) {
    float sum = 0.0;
    for (int i = 0; i < col; i++) {
      float val = array[index * col + i];
      sum += val * val;
    }
    norm[index] = sum;
  }
}

__global__ void broadcast_points(float *A_norm, float *B_norm, int m, int n,
                                 float *C) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  __shared__ float Ads[TILE_WIDTH];
  __shared__ float Bds[TILE_WIDTH];

  // Load values into shared memory
  if (tx == 0 && col < n) {
    Ads[ty] = A_norm[row];
  }

  if (ty == 0 && row < m) {
    Bds[tx] = B_norm[col];
  }

  __syncthreads();

  // Each (i, j) element in C need to add A_norm[i] and B_norm[j]
  if (row < m && col < n) {
    float val = C[row * n + col] + Ads[ty] + Bds[tx];
    C[row * n + col] = sqrtf(val);
  }
}

__global__ void insertion_select_col_kernel(const float *dlist, int *ilist,
                                            float *dqueue, int m, int n,
                                            int k) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  ilist[i] = 0;
  dqueue[i] = dlist[i];
  for (int j = 1; j < m; j++) {
    float cur_dist = dlist[j * n + i];
    if (j >= k && (dqueue[(k - 1) * n + i] <= cur_dist)) {
      continue;
    }
    int index = (j < k) ? j : k;
    while (index > 0 && (dqueue[(index - 1) * n + i] > cur_dist)) {
      if (index != k) {
        dqueue[index * n + i] = dqueue[(index - 1) * n + i];
        ilist[index * n + i] = ilist[(index - 1) * n + i];
      }
      index--;
    }
    dqueue[index * n + i] = cur_dist;
    ilist[index * n + i] = j;
  }
}

void euclidean_distance(const float *A, const float *B, float *D, int m,
                        int n, int k) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = -2.0;
  float beta = 0.0;

  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B, k, A, k,
              &beta, D, n);

  cublasDestroy(handle);

  float *rnorm, *qnorm;
  cudaMalloc((void **)&rnorm, sizeof(float) * m);
  cudaMalloc((void **)&qnorm, sizeof(float) * n);

  dim3 dimGrid(ceil(m / 256.0), 1, 1);
  dim3 dimBlock(256, 1, 1);

  compute_squared_norm<<<dimGrid, dimBlock>>>(A, m, k, rnorm);

  dimGrid.x = ceil(n / 256.0);
  compute_squared_norm<<<dimGrid, dimBlock>>>(B, n, k, qnorm);

  dimBlock.x = 32;
  dimBlock.y = 32;
  dimGrid.x = ceil(n / 32.0);
  dimGrid.y = ceil(m / 32.0);

  broadcast_points<<<dimGrid, dimBlock>>>(rnorm, qnorm, m, n, D);
  cudaFree(rnorm);
  cudaFree(qnorm);
}

void insertion_select_col(const float *dlist, int *ilist, int m, int n, int k){
  dim3 dimGrid(ceil(n / 256.0), 1, 1);
  dim3 dimBlock(256, 1, 1);

  float *dqueue;
  cudaMalloc((void **)&dqueue, n * k * sizeof(float));
  insertion_select_col_kernel<<<dimGrid, dimBlock>>>(dlist, ilist, dqueue, m, n, k);
  cudaFree(dqueue);
}

void knn_impl(const Tensor &ref, const Tensor &labels, const Tensor &query,
              int k, Tensor &result) {
  int num_of_ref = ref.dim(0);
  int num_of_query = query.dim(0);
  int num_of_feature = ref.dim(1);
  result.resize({num_of_query, 1});

  ScalarType sc_ref_type = ref.scalar_type();
  ScalarType sc_query_type = query.scalar_type();
  HICE_CHECK_EQ(sc_ref_type, sc_query_type)
      << "Both scalar types of arguments to knn must be equal";
  const float* ref_data = ref.data<float>();
  const float* query_data = query.data<float>();
  Tensor labels_cpu = labels.to(kCPU);
  const int* label_data = labels_cpu.data<int>();
  Tensor distance({num_of_ref, num_of_query},
                  device(ref.device()).dtype(kFloat));
  euclidean_distance(ref_data, query_data, distance.mutable_data<float>(), num_of_ref, num_of_query,
                     num_of_feature);
  // TensorPrinter tp;
  // tp.print(distance);
  // top k
  Tensor ilist({k, num_of_query}, device(ref.device()).dtype(kInt32));
  insertion_select_col(distance.data<float>(), ilist.mutable_data<int>(), num_of_ref, num_of_query, k);
  // tp.print(ilist);
  // choose labels
  Tensor ilist_cpu = ilist.to(kCPU);
  Tensor result_cpu = result.to(kCPU);
  auto result_data = result_cpu.mutable_data<int>();
  for (int i = 0; i < num_of_query; i++) {
    std::unordered_map<int, int> count;
    int label_ = -1;
    for (int j = 0; j < k; j++) {
      int index = label_data[ilist_cpu.data<int>()[j * num_of_query + i]];
      int maxCount = 0;
      std::unordered_map<int, int>::iterator it = count.find(index);
      if (it != count.end()) {
        it->second++;
      } else {
        count.insert(std::make_pair(index, 1));
      }
      if (count[index] > maxCount) {
        maxCount = count[index];
        label_ = index;
      }
    }
    result_data[i] = label_;
  }
  result = result_cpu.to(kCUDA);
}

}  // namespace

HICE_REGISTER_KERNEL(knn_dispatcher, &knn_impl,
                     {kCUDA, kDense},  // ref
                     {kCUDA, kDense},  // labels
                     {kCUDA, kDense},  // query
                     {kCUDA, kDense}   // result
);

}  // namespace hice
