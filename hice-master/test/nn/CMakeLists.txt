set(module_name nn)

list(APPEND hice_test_${module_name}_srcs_cpu 
  ${module_name}/activation_test.cpp
  ${module_name}/batch_norm_dense_test.cpp
  ${module_name}/conv_test.cpp
  ${module_name}/cross_entropy_test.cpp
  ${module_name}/ctc_loss_test.cpp
  ${module_name}/dropout_dense_test.cpp
  ${module_name}/l1_loss_test.cpp
  ${module_name}/mse_loss_test.cpp
  ${module_name}/nll_loss_test.cpp
  ${module_name}/pooling_dense_test.cpp
  ${module_name}/relu_family_test.cpp
  ${module_name}/smooth_l1_loss_test.cpp
  ${module_name}/softmax_cross_entropy_test.cpp
  ${module_name}/softmax_test.cpp
  ${module_name}/sparse_cross_entropy_test.cpp
  ${module_name}/sparse_softmax_cross_entropy_test.cpp
)

set(hice_test_${module_name}_srcs_cpu  ${hice_test_${module_name}_srcs_cpu} PARENT_SCOPE)