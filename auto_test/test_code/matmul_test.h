#pragma once

#include <ctime>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include "src/math/matmul.h"
#include <math.h>
}

namespace aitisa_api {

template <typename InterfaceType>
class MatmulTest : public ::testing::Test{
public:
  MatmulTest():
    input0(/*ndim1*/1, /*dims1*/{10}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/1, /*dims2*/{10}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input1(/*ndim1*/2, /*dims1*/{1995,2020}, /*dtype1=double*/9,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0,
            /*ndim2*/2, /*dims2*/{2020,2018}, /*dtype2=double*/9, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input2(/*ndim1*/1, /*dims1*/{10}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/2, /*dims2*/{10,5}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input3(/*ndim1*/2, /*dims1*/{10,5}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/1, /*dims2*/{5}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input4(/*ndim1*/1, /*dims1*/{3}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/5, /*dims2*/{2,2,4,3,2}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input5(/*ndim1*/5, /*dims1*/{2,2,4,2,3}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/1, /*dims2*/{3}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0),
    input6(/*ndim1*/3, /*dims1*/{2,4,3}, /*dtype1=float*/8,  
            /*device1=cpu*/0, /*data1*/nullptr, /*len1*/0, 
            /*ndim2*/4, /*dims2*/{3,2,3,2}, /*dtype2=float*/8, 
            /*device2=cpu*/0, /*data2*/nullptr, /*len2*/0){
    input[0] = &input0;
    input[1] = &input1;
    input[2] = &input2;
    input[3] = &input3;
    input[4] = &input4;
    input[5] = &input5;
    input[6] = &input6;
    ninput = 7;
    for(int i=0; i<ninput; i++){
      unsigned int input_nelem1 = 1;
      unsigned int input_nelem2 = 1;
      for(unsigned int j=0; j<input[i]->ndim1(); j++){
        input_nelem1 *= input[i]->dims1()[j];
      }
      for(unsigned int j=0; j<input[i]->ndim2(); j++){
        input_nelem2 *= input[i]->dims2()[j];
      }
      unsigned int input_len1 = input_nelem1 * elem_size(input[i]->dtype1());
      unsigned int input_len2 = input_nelem2 * elem_size(input[i]->dtype2());
      void *input_data1 = (void*) new char[input_len1];
      void *input_data2 = (void*) new char[input_len2];
      if(i == 1){ 
        random_assign(input_data1, input_len1, input[i]->dtype1());
        random_assign(input_data2, input_len2, input[i]->dtype2());
      }else{
        natural_assign(input_data1, input_len1, input[i]->dtype1());
        natural_assign(input_data2, input_len2, input[i]->dtype2());
      }
      input[i]->set_data1(input_data1, input_len1);
      input[i]->set_data2(input_data2, input_len2);
    }
  }
  virtual ~MatmulTest(){}
  using InputType = Binary_Input;
  using UserInterface = InterfaceType;
  static void aitisa_kernel(AITISA_Tensor in1, AITISA_Tensor in2, AITISA_Tensor *out){
    aitisa_matmul(in1, in2, out);
  }
  // inputs
  Binary_Input input0;
  Binary_Input input1;
  Binary_Input input2;
  Binary_Input input3;
  Binary_Input input4;
  Binary_Input input5;
  Binary_Input input6;
  Binary_Input *input[7] = {&input0, &input1, &input2, &input3, &input4, &input5, &input6};
  std::string input0_name = "Natural Float CPU with Dims{10} and Dims{10}";
  std::string input1_name = "Random Double CPU with Dims{199,202} and Dims{202,201}";
  std::string input2_name = "Natural Float CPU with Dims{10} and Dims{10,5}";
  std::string input3_name = "Natural Float CPU with Dims{10,5} and Dims{5}";
  std::string input4_name = "Natural Float CPU with Dims{3} and Dims{2,2,4,3,2}";
  std::string input5_name = "Natural Float CPU with Dims{2,2,4,2,3} and Dims{3}";
  std::string input6_name = "Natural Float CPU with Dims{2,4,3} and Dims{3,2,3,2}";
  std::string *input_name[7] = {&input0_name, &input1_name, &input2_name, &input3_name, 
                                &input4_name, &input5_name, &input6_name};
  int ninput = 7;
};
TYPED_TEST_CASE_P(MatmulTest);

TYPED_TEST_P(MatmulTest, SevenTests){
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;
  for(int i=0; i<this->ninput; i++){
    std::clock_t aitisa_start, aitisa_end, user_start, user_end;
    double aitisa_time, user_time;
    int64_t aitisa_result_ndim, user_result_ndim;
    int64_t *aitisa_result_dims=nullptr, *user_result_dims=nullptr;
    float *aitisa_result_data=nullptr, *user_result_data=nullptr;
    unsigned int aitisa_result_len, user_result_len;
    AITISA_Tensor aitisa_tensor1, aitisa_tensor2, aitisa_result;
    AITISA_DataType aitisa_result_dtype;
    AITISA_Device aitisa_result_device;
    UserTensor user_tensor1, user_tensor2, user_result;
    UserDataType user_result_dtype;
    UserDevice user_result_device;
    // aitisa
    AITISA_DataType aitisa_dtype1 = aitisa_int_to_dtype(this->input[i]->dtype1());
    AITISA_DataType aitisa_dtype2 = aitisa_int_to_dtype(this->input[i]->dtype2());
    AITISA_Device aitisa_device1 = aitisa_int_to_device(0); // cpu supoorted only
    AITISA_Device aitisa_device2 = aitisa_int_to_device(0); // cpu supported only
    aitisa_create(aitisa_dtype1, aitisa_device1, this->input[i]->dims1(), this->input[i]->ndim1(), 
                  (void*)(this->input[i]->data1()), this->input[i]->len1(), &aitisa_tensor1);
    aitisa_create(aitisa_dtype2, aitisa_device2, this->input[i]->dims2(), this->input[i]->ndim2(), 
                  (void*)(this->input[i]->data2()), this->input[i]->len2(), &aitisa_tensor2);
    aitisa_start = std::clock();
    aitisa_matmul(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
    aitisa_end = std::clock();
    aitisa_time = 1000.0 * (aitisa_end - aitisa_start) / static_cast<double>(CLOCKS_PER_SEC);
    aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device, &aitisa_result_dims, 
                   &aitisa_result_ndim, (void**)&aitisa_result_data, &aitisa_result_len);
    // user
    UserDataType user_dtype1 = UserFuncs::user_int_to_dtype(this->input[i]->dtype1());
    UserDataType user_dtype2 = UserFuncs::user_int_to_dtype(this->input[i]->dtype2());
    UserDevice user_device1 = UserFuncs::user_int_to_device(this->input[i]->device1());
    UserDevice user_device2 = UserFuncs::user_int_to_device(this->input[i]->device2());
    UserFuncs::user_create(user_dtype1, user_device1, this->input[i]->dims1(), 
                           this->input[i]->ndim1(), this->input[i]->data1(),
                           this->input[i]->len1(), &user_tensor1);
    UserFuncs::user_create(user_dtype2, user_device2, this->input[i]->dims2(), 
                           this->input[i]->ndim2(), this->input[i]->data2(), 
                           this->input[i]->len2(), &user_tensor2);
    user_start = std::clock();
    UserFuncs::user_matmul(user_tensor1, user_tensor2, &user_result);
    user_end = std::clock();
    user_time = 1000.0 * (user_end - user_start) / static_cast<double>(CLOCKS_PER_SEC);
    UserFuncs::user_resolve(user_result, &user_result_dtype, &user_result_device, 
                            &user_result_dims, &user_result_ndim, 
                            (void**)&user_result_data, &user_result_len);
    // compare
    int64_t tensor_size = 1;
    ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
    if(i == 1){ // CUDA
      ASSERT_EQ(
        /*CUDA*/0, UserFuncs::user_device_to_int(user_result_device));
    }else{ // CPU
      ASSERT_EQ(aitisa_device_to_int(aitisa_result_device), 
                UserFuncs::user_device_to_int(user_result_device));
    }
    ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype), 
              UserFuncs::user_dtype_to_int(user_result_dtype));
    for(int64_t j=0; j<aitisa_result_ndim; j++){
      tensor_size *= aitisa_result_dims[j];
      ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
    }
    ASSERT_EQ(aitisa_result_len, user_result_len);
    if(i == 1){ // Double
      double *aitisa_data = (double*)aitisa_result_data;
      double *user_data = (double*)user_result_data;
      for(int64_t j=0; j<tensor_size; j++){
        ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
      }
    }else{ // Float
      for(int64_t j=0; j<tensor_size; j++){
        ASSERT_TRUE(abs(aitisa_result_data[j] - user_result_data[j]) < 1e-3);
      }
    }
    // print result of test
    std::cout<< /*GREEN <<*/ "[ Matmul sample"<< i << " / " 
             << *(this->input_name[i]) << " ] " << /*RESET <<*/ std::endl;
    std::cout<< /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time << " ms" << std::endl;
    std::cout<< /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time << " ms" << std::endl;
  }
}
REGISTER_TYPED_TEST_CASE_P(MatmulTest, SevenTests);

Sample<Binary_Input> get_sample_matmul(int sample_num){
  return get_binary_sample<MatmulTest<void>>(sample_num);
}

#define GET_SAMPLE_MATMUL(SAMPLE, NUM)                                                    \
  aitisa_api::Sample<aitisa_api::Binary_Input> SAMPLE;                                    \
  aitisa_api::get_binary_sample<aitisa_api::MatmulTest<void>>(SAMPLE, NUM);

#define REGISTER_MATMUL(MATMUL_FUNC, MATMUL)                                              \
  class Matmul : public Basic {                                                           \
  public:                                                                                 \
    static void user_matmul(UserTensor tensor1, UserTensor tensor2, UserTensor* result){  \
      typedef std::function<void(UserTensor,UserTensor,UserTensor*)> matmul_func;         \
      auto func_args_num = aitisa_api::function_traits<MATMUL_FUNC>::nargs;               \
      auto args_num = aitisa_api::function_traits<matmul_func>::nargs;                    \
        if(func_args_num != args_num){                                                    \
            throw std::invalid_argument(                                                  \
                "Incorrect parameter numbers: expected " +                                \
                std::to_string(args_num) +                                                \
                " arguments but got " +                                                   \
                std::to_string(func_args_num));                                           \
        }                                                                                 \
        if(!std::is_same<                                                                 \
            std::remove_cv<aitisa_api::function_traits<matmul_func>::result_type>::type,  \
                aitisa_api::function_traits<MATMUL_FUNC>::result_type>::value){           \
            throw std::invalid_argument("Incorrect return type: type mismatch at return");\
            }                                                                             \
        aitisa_api::TypeCompare<                                                          \
            aitisa_api::function_traits<matmul_func>::nargs,                              \
            matmul_func,                                                                  \
            MATMUL_FUNC                                                                   \
        >();                                                                              \
    MATMUL(tensor1, tensor2, result);                                                     \
    }                                                                                     \
  };                                                                                      \
  namespace aitisa_api{                                                                   \
    INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, MatmulTest, Matmul);                        \
  }

} // namespace aitisa_api