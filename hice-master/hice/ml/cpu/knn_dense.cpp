#include "hice/ml/knn.h"

namespace hice {

namespace {

template <typename scalar_t>
void euclidean_distance(const scalar_t* A, const scalar_t* B, float* D, int m,
                        int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float d = 0.0;
      for (int q = 0; q < k; q++) {
        float diff = A[i * k + q] - B[j * k + q];
        d += diff * diff;
      }
      D[i * n + j] = sqrtf(d);
    }
  }
}

void insertion_select_col(const float* dlist, int* ilist, int m, int n, int k) {
  float* dqueue = new float[k * n];
  for (int i = 0; i < n; i++) {
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
  delete[] dqueue;
}

void knn_impl(const Tensor& ref, const Tensor& labels, const Tensor& query,
              int k, Tensor& result) {
  int num_of_ref = ref.dim(0);
  int num_of_query = query.dim(0);
  int num_of_feature = ref.dim(1);
  result.resize({num_of_query, 1});

  ScalarType sc_ref_type = ref.scalar_type();
  ScalarType sc_query_type = query.scalar_type();
  HICE_CHECK_EQ(sc_ref_type, sc_query_type)
      << "Both scalar types of arguments to knn must be equal";
  HICE_DISPATCH_ALL_TYPES(sc_ref_type, "KNN", [&]() {
    // calculate the distance

    const scalar_t* ref_data = ref.data<scalar_t>();
    const scalar_t* query_data = query.data<scalar_t>();
    const int* label_data = labels.data<int>();
    float* distance = new float[num_of_ref * num_of_query];
    euclidean_distance(ref_data, query_data, distance, num_of_ref, num_of_query,
                       num_of_feature);
    // for (int i = 0;i < num_of_ref;i++){
    //   for (int j = 0;j < num_of_query;j++){
    //     std::cout << distance[i * num_of_query + j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // top k
    int* ilist = new int[k * num_of_query];
    insertion_select_col(distance, ilist, num_of_ref, num_of_query, k);
    // for (int i = 0; i < k; i++) {
    //   for (int j = 0; j < num_of_query; j++) {
    //     std::cout << ilist[i * num_of_query + j] << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // choose labels
    auto result_data = result.mutable_data<int>();
    for (int i = 0; i < num_of_query; i++) {
      std::unordered_map<int, int> count;
      int label_ = -1;
      for (int j = 0; j < k; j++) {
        int index = label_data[ilist[j * num_of_query + i]];
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

    delete[] distance;
    delete[] ilist;
  });
}

}  // namespace

HICE_REGISTER_KERNEL(knn_dispatcher, &knn_impl, {kCPU, kDense},  // ref
                     {kCPU, kDense},                             // labels
                     {kCPU, kDense},                             // query
                     {kCPU, kDense}                              // result
);

}  // namespace hice
