set(module_name ml)

list(APPEND hice_test_${module_name}_srcs_cpu 
  ${module_name}/knn_dense_test.cpp
  # ${module_name}/svm_sparse_test.cpp
  # ${module_name}/svm_test.cpp
  ${module_name}/dataset_dense_test.cpp
)

set(hice_test_${module_name}_srcs_cpu  ${hice_test_${module_name}_srcs_cpu} PARENT_SCOPE)