#pragma once

#include <ctime>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include <math.h>
#include "src/math/matmul.h"
}

namespace aitisa_api {

template <typename InterfaceType>
class MatmulTest : public ::testing::Test {
 public:
  MatmulTest() { fetch_test_data("matmul", matmul_inputs, matmul_inputs_name); }
  ~MatmulTest() override = default;

  static void aitisa_kernel(AITISA_Tensor in1, AITISA_Tensor in2,
                            AITISA_Tensor* out) {
    aitisa_matmul(in1, in2, out);
  }

  int fetch_test_data(const char* path, std::vector<Binary_Input>& inputs,
                      std::vector<std::string>& inputs_name) {

    config_t cfg;
    config_setting_t* setting;
    const char* str;
    config_init(&cfg);

    /* Read the file. If there is an error, report it and exit. */
    if (!config_read_file(&cfg, CONFIG_FILE)) {
      fprintf(stderr, "%s:%d - %s\n", config_error_file(&cfg),
              config_error_line(&cfg), config_error_text(&cfg));
      config_destroy(&cfg);
      return (EXIT_FAILURE);
    }

    setting = config_lookup(&cfg, path);

    if (setting != nullptr) {
      int count = config_setting_length(setting);

      for (int i = 0; i < count; ++i) {
        config_setting_t* test = config_setting_get_elem(setting, i);
        config_setting_t* dims1_setting = config_setting_lookup(test, "dims1");
        config_setting_t* dims2_setting = config_setting_lookup(test, "dims2");

        int64_t ndim1, ndim2;
        std::vector<int64_t> dims1, dims2;
        int dtype1, device1, len1, dtype2, device2, len2;
        const char* input_name;

        if (!config_setting_lookup_int64(
                test, "ndim1", reinterpret_cast<long long int*>(&ndim1))) {
          fprintf(stderr, "No 'ndim1' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_int64(
                test, "ndim2", reinterpret_cast<long long int*>(&ndim2))) {
          fprintf(stderr, "No 'ndim2' in test case %d from %s.\n", i, path);
          continue;
        }
        for (int j = 0; j < ndim1; ++j) {
          int64_t input = config_setting_get_int_elem(dims1_setting, j);
          dims1.push_back(input);
        }
        for (int j = 0; j < ndim2; ++j) {
          int64_t input = config_setting_get_int_elem(dims2_setting, j);
          dims2.push_back(input);
        }
        if (!config_setting_lookup_int(test, "dtype1", &dtype1)) {
          fprintf(stderr, "No 'dtype1' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_int(test, "dtype2", &dtype2)) {
          fprintf(stderr, "No 'dtype2' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_int(test, "device1", &device1)) {
          fprintf(stderr, "No 'device1' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_int(test, "device2", &device2)) {
          fprintf(stderr, "No 'device2' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_int(test, "len1", &len1)) {
          fprintf(stderr, "No 'len1' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_int(test, "len2", &len2)) {
          fprintf(stderr, "No 'len2' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_string(test, "input_name", &input_name)) {
          fprintf(stderr, "No 'input_name' in test case %d from %s.\n", i,
                  path);
          continue;
        }

        Binary_Input tmp(ndim1, dims1, dtype1, device1, nullptr, len1, ndim2,
                         dims2, dtype2, device2, nullptr, len2);

        inputs.push_back(std::move(tmp));
        inputs_name.emplace_back(input_name);
      }
    }

    for (auto& input : inputs) {
      unsigned int input_nelem1 = 1;
      unsigned int input_nelem2 = 1;
      for (unsigned int j = 0; j < input.ndim1(); j++) {
        input_nelem1 *= input.dims1()[j];
      }
      for (unsigned int j = 0; j < input.ndim2(); j++) {
        input_nelem2 *= input.dims2()[j];
      }
      unsigned int input_len1 = input_nelem1 * elem_size(input.dtype1());
      unsigned int input_len2 = input_nelem2 * elem_size(input.dtype2());
      void* input_data1 = (void*)new char[input_len1];
      void* input_data2 = (void*)new char[input_len2];
      //      if ( i == 1) {
      //        random_assign(input_data1, input_len1, input[i]->dtype1());
      //        random_assign(input_data2, input_len2, input[i]->dtype2());
      //      } else {
      natural_assign(input_data1, input_len1, input.dtype1());
      natural_assign(input_data2, input_len2, input.dtype2());
      //      }
      input.set_data1(input_data1, input_len1);
      input.set_data2(input_data2, input_len2);
    }

    config_destroy(&cfg);
    return (EXIT_SUCCESS);
  }
  using InputType = Binary_Input;
  using UserInterface = InterfaceType;
  std::vector<Binary_Input> matmul_inputs;
  std::vector<std::string> matmul_inputs_name;
  std::map<std::string, int> test_case = {{"matmul", 0}};
};
TYPED_TEST_CASE_P(MatmulTest);

TYPED_TEST_P(MatmulTest, SevenTests) {
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;

  auto test = [](std::vector<Binary_Input>&& inputs,
                 std::vector<std::string>&& inputs_name,
                 const std::string& test_case_name, int test_case_index) {
    for (int i = 0; i < inputs.size(); i++) {
      std::clock_t aitisa_start, aitisa_end, user_start, user_end;
      double aitisa_time, user_time;
      int64_t aitisa_result_ndim, user_result_ndim;
      int64_t *aitisa_result_dims = nullptr, *user_result_dims = nullptr;
      float *aitisa_result_data = nullptr, *user_result_data = nullptr;
      unsigned int aitisa_result_len, user_result_len;
      AITISA_Tensor aitisa_tensor1, aitisa_tensor2, aitisa_result;
      AITISA_DataType aitisa_result_dtype;
      AITISA_Device aitisa_result_device;
      UserTensor user_tensor1, user_tensor2, user_result;
      UserDataType user_result_dtype;
      UserDevice user_result_device;
      // aitisa
      AITISA_DataType aitisa_dtype1 = aitisa_int_to_dtype(inputs[i].dtype1());
      AITISA_DataType aitisa_dtype2 = aitisa_int_to_dtype(inputs[i].dtype2());
      AITISA_Device aitisa_device1 =
          aitisa_int_to_device(0);  // cpu supoorted only
      AITISA_Device aitisa_device2 =
          aitisa_int_to_device(0);  // cpu supported only
      aitisa_create(aitisa_dtype1, aitisa_device1, inputs[i].dims1(),
                    inputs[i].ndim1(), (void*)(inputs[i].data1()),
                    inputs[i].len1(), &aitisa_tensor1);
      aitisa_create(aitisa_dtype2, aitisa_device2, inputs[i].dims2(),
                    inputs[i].ndim2(), (void*)(inputs[i].data2()),
                    inputs[i].len2(), &aitisa_tensor2);
      aitisa_start = std::clock();
      aitisa_matmul(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
      aitisa_end = std::clock();
      aitisa_time = 1000.0 * (aitisa_end - aitisa_start) /
                    static_cast<double>(CLOCKS_PER_SEC);
      aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device,
                     &aitisa_result_dims, &aitisa_result_ndim,
                     (void**)&aitisa_result_data, &aitisa_result_len);
      // user
      UserDataType user_dtype1 =
          UserFuncs::user_int_to_dtype(inputs[i].dtype1());
      UserDataType user_dtype2 =
          UserFuncs::user_int_to_dtype(inputs[i].dtype2());
      UserDevice user_device1 =
          UserFuncs::user_int_to_device(inputs[i].device1());
      UserDevice user_device2 =
          UserFuncs::user_int_to_device(inputs[i].device2());
      UserFuncs::user_create(user_dtype1, user_device1, inputs[i].dims1(),
                             inputs[i].ndim1(), inputs[i].data1(),
                             inputs[i].len1(), &user_tensor1);
      UserFuncs::user_create(user_dtype2, user_device2, inputs[i].dims2(),
                             inputs[i].ndim2(), inputs[i].data2(),
                             inputs[i].len2(), &user_tensor2);
      user_start = std::clock();
      UserFuncs::user_matmul(user_tensor1, user_tensor2, &user_result);
      user_end = std::clock();
      user_time = 1000.0 * (user_end - user_start) /
                  static_cast<double>(CLOCKS_PER_SEC);
      UserFuncs::user_resolve(user_result, &user_result_dtype,
                              &user_result_device, &user_result_dims,
                              &user_result_ndim, (void**)&user_result_data,
                              &user_result_len);
      // compare
      int64_t tensor_size = 1;
      ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
      if (i == 1) {  // CUDA
        ASSERT_EQ(
            /*CUDA*/ 0, UserFuncs::user_device_to_int(user_result_device));
      } else {  // CPU
        ASSERT_EQ(aitisa_device_to_int(aitisa_result_device),
                  UserFuncs::user_device_to_int(user_result_device));
      }
      ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype),
                UserFuncs::user_dtype_to_int(user_result_dtype));
      for (int64_t j = 0; j < aitisa_result_ndim; j++) {
        tensor_size *= aitisa_result_dims[j];
        ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
      }
      ASSERT_EQ(aitisa_result_len, user_result_len);
      //      if (i == 1) {  // Double
      //        auto* aitisa_data = (double*)aitisa_result_data;
      //        auto* user_data = (double*)user_result_data;
      //        for (int64_t j = 0; j < tensor_size; j++) {
      //          ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
      //        }
      //      } else {  // Float
      for (int64_t j = 0; j < tensor_size; j++) {
        ASSERT_TRUE(abs(aitisa_result_data[j] - user_result_data[j]) < 1e-3);
      }
      //      }
      // print result of test
      std::cout << /*GREEN <<*/ "[ " << test_case_name << " sample" << i
                << " / " << inputs_name[i] << " ] " << /*RESET <<*/ std::endl;
      std::cout << /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time
                << " ms" << std::endl;
      std::cout << /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time
                << " ms" << std::endl;
    }
  };

  test(std::move(this->matmul_inputs), std::move(this->matmul_inputs_name),
       "matmul", this->test_case["matmul"]);
}
REGISTER_TYPED_TEST_CASE_P(MatmulTest, SevenTests);

//Sample<Binary_Input> get_sample_matmul(int sample_num) {
//  return get_binary_sample<MatmulTest<void>>(sample_num);
//}

#define GET_SAMPLE_MATMUL(SAMPLE, NUM)                 \
  aitisa_api::Sample<aitisa_api::Binary_Input> SAMPLE; \
  aitisa_api::get_binary_sample<aitisa_api::MatmulTest<void>>(SAMPLE, NUM);

#define REGISTER_MATMUL(MATMUL_FUNC, MATMUL)                                   \
  class Matmul : public Basic {                                                \
   public:                                                                     \
    static void user_matmul(UserTensor tensor1, UserTensor tensor2,            \
                            UserTensor* result) {                              \
      typedef std::function<void(UserTensor, UserTensor, UserTensor*)>         \
          matmul_func;                                                         \
      auto func_args_num = aitisa_api::function_traits<MATMUL_FUNC>::nargs;    \
      auto args_num = aitisa_api::function_traits<matmul_func>::nargs;         \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<                                                       \
              std::remove_cv<aitisa_api::function_traits<                      \
                  matmul_func>::result_type>::type,                            \
              aitisa_api::function_traits<MATMUL_FUNC>::result_type>::value) { \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<aitisa_api::function_traits<matmul_func>::nargs, \
                              matmul_func, MATMUL_FUNC>();                     \
      MATMUL(tensor1, tensor2, result);                                        \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, MatmulTest, Matmul);               \
  }

}  // namespace aitisa_api