#include "hice/core/tensor_printer.h"
#include "hice/basic/factories.h"
#include "hice/basic/copy.h"
#include "hice/core/layout_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace hice {

using ::testing::ElementsAre;
using ::testing::ContainerEq;
using ::testing::IsEmpty;

template<typename TScalarType1, typename TScalarType2>
void TENSOR_EXPECT_EQ(Tensor& tensor1, Tensor& tensor2) {
  EXPECT_EQ(tensor1.size(), tensor2.size());
  EXPECT_EQ(tensor1.offset(), tensor2.offset());
  // EXPECT_EQ(tensor1.data_type(), tensor2.data_type());
  // EXPECT_EQ(tensor1.device(), tensor2.device());
  EXPECT_EQ(tensor1.shape(), tensor2.shape());
  EXPECT_EQ(tensor1.strides(), tensor2.strides());
  Tensor tensor1_new = tensor1.device_type() == kCPU ? tensor1 : tensor1.to(kCPU);
  Tensor tensor2_new = tensor2.device_type() == kCPU ? tensor2 : tensor2.to(kCPU);
  auto size = tensor1.size();
  for(int i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(tensor1_new.data<TScalarType1>()[i], 
                    tensor2_new.data<TScalarType2>()[i]);
  }
}


template<typename scalar_t_src, typename scalar_t_dst>
void My_test(hice::DeviceType device_type_src,
            hice::ScalarType scalar_type_src,
            hice::DeviceType device_type_dst,
            hice::ScalarType scalar_type_dst){
  TensorPrinter tp;
  TensorOptions options_src = device(device_type_src).dtype(scalar_type_src);
  TensorOptions options_dst = device(device_type_dst).dtype(scalar_type_dst);

  Tensor src = rand_uniform({}, 1.0, 10.0, options_src);
  Tensor dst = empty({}, options_dst);
  copy(src, dst);
  TENSOR_EXPECT_EQ<scalar_t_src, scalar_t_dst>(src, dst);

  src = rand_uniform({0}, 1.0, 10.0, options_src);
  dst = empty({0}, options_dst);
  copy(src, dst);
  TENSOR_EXPECT_EQ<scalar_t_src, scalar_t_dst>(dst, src);

  src = rand_uniform({1}, 1.0, 10.0, options_src);
  dst = empty({1}, options_dst);
  copy(src, dst);
  TENSOR_EXPECT_EQ<scalar_t_src, scalar_t_dst>(src, dst);

  src = rand_uniform({2, 3}, 1.0, 10.0, options_src);
  dst = empty({2, 3}, options_dst);
  copy(src, dst);
  TENSOR_EXPECT_EQ<scalar_t_src, scalar_t_dst>(src, dst);

  // std::cout<<"==src=="<<std::endl;
  // tp.print(src);
  // std::cout<<"==dst=="<<std::endl;
  // tp.print(dst);

  src = rand_uniform({4, 5, 6}, 1.0, 10.0, options_src);
  dst = empty({4, 5, 6}, options_dst);
  copy(src, dst);
  TENSOR_EXPECT_EQ<scalar_t_src, scalar_t_dst>(src, dst);

  // parallel if openmp enabled
  src = rand_uniform({5, 4, 50, 60}, 1.0, 10.0, options_src);
  dst = empty({5, 4, 50, 60}, options_dst);
  copy(src, dst);
  TENSOR_EXPECT_EQ<scalar_t_src, scalar_t_dst>(src, dst);
}

// TEST(CopyTestSuite, CPU_to_CPU) {
//   // same type
//   My_test<float, float>(kCPU, kFloat,
//                                 kCPU, kFloat);
//   // cross type
//   My_test<float, double>(kCPU, kFloat,
//                                  kCPU, kDouble);
// }

// TEST(CopyTestSuite, CPU_to_CUDA) {
//   // same type
//   My_test<float, float>(kCPU, kFloat,
//                                 kCUDA, kFloat);
//   // cross type
//   My_test<float, double>(kCPU, kFloat,
//                                  kCUDA, kDouble);
// }

// TEST(CopyTestSuite, CUDA_to_CUDA) {
//   // same type
//   My_test<float, float>(kCUDA, kFloat,
//                                 kCUDA, kFloat);
//   // cross type
//   My_test<float, double>(kCUDA, kFloat,
//                                  kCUDA, kDouble);
// }

// TEST(CopyTestSuite, CUDA_to_CPU) {
//   // same type
//   My_test<float, float>(kCUDA, kFloat,
//                                 kCPU, kFloat);
//   // cross type
//   My_test<float, double>(kCUDA, kFloat,
//                                  kCPU, kDouble);
// }

TEST(CopyTestSuite, CPUNonContiguous) {
  using scalar_t = float; 
  auto scalar_type = kFloat;
  auto device_type = kCUDA;

  TensorPrinter tp;
  TensorOptions options_cgs = device(device_type).dtype(scalar_type);
  TensorOptions options_noncgs = 
    device(device_type).dtype(scalar_type).layout(LayoutUtil::make_layout({0, 2, 1}));

  Tensor src_cgs = rand_uniform({2, 3, 2}, 1.0, 10.0, options_cgs);
  Tensor src_noncgs = rand_uniform({2, 3, 2}, 1.0, 10.0, options_noncgs);
  Tensor dst_cgs = rand_uniform({2, 3, 2}, 1.0, 10.0, options_cgs);

  // std::cout<<"==src_cgs=="<<std::endl;
  // tp.print(src_cgs);
  // std::cout<<"==src_noncgs=="<<std::endl;
  // tp.print(src_noncgs);
  // std::cout<<"==dst_cgs=="<<std::endl;
  // tp.print(dst_cgs);

  copy(src_cgs, src_noncgs);
  copy(src_noncgs, dst_cgs);
  TENSOR_EXPECT_EQ<scalar_t, scalar_t>(src_cgs, dst_cgs);

  // std::cout<<"==src_cgs=="<<std::endl;
  // tp.print(src_cgs);
  // std::cout<<"==src_noncgs=="<<std::endl;
  // tp.print(src_noncgs);
  // std::cout<<"==dst_cgs=="<<std::endl;
  // tp.print(dst_cgs);
}

}  // namespace hice
