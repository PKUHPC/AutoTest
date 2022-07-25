#pragma once

#include <ctime>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include <math.h>
#include "src/math/binary_op.h"
}

namespace aitisa_api {

template <typename InterfaceType>
class BinaryOPTest : public ::testing::Test {
 public:
  BinaryOPTest() {
    fetch_test_data("binary_op.add", add_inputs, add_inputs_name,
                    test_case["add"]);
    fetch_test_data("binary_op.sub", sub_inputs, sub_inputs_name,
                    test_case["sub"]);
    fetch_test_data("binary_op.mul", mul_inputs, mul_inputs_name,
                    test_case["mul"]);
    fetch_test_data("binary_op.div", div_inputs, div_inputs_name,
                    test_case["div"]);
  }
  virtual ~BinaryOPTest() {}
  int fetch_test_data(const char* path, std::vector<Binary_Input>& inputs,
                      std::vector<std::string>& inputs_name,
                      int test_case_index) {

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
        for (int j = 0; j < ndim1; ++j) {
          int64_t input = config_setting_get_int_elem(dims1_setting, j);
          dims1.push_back(input);
        }
        for (int j = 0; j < ndim2; ++j) {
          int64_t input = config_setting_get_int_elem(dims2_setting, j);
          dims2.push_back(input);
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
      auto input_data1 = (void*)new char[input_len1];
      auto input_data2 = (void*)new char[input_len2];
      if (test_case_index == 1) {
        random_assign(input_data1, input_len1, input.dtype1());
        random_assign(input_data2, input_len2, input.dtype2());
      } else {
        natural_assign(input_data1, input_len1, input.dtype1());
        natural_assign(input_data2, input_len2, input.dtype2());
      }
      input.set_data1(input_data1, input_len1);
      input.set_data2(input_data2, input_len2);
    }

    config_destroy(&cfg);
    return (EXIT_SUCCESS);
  }

  using InputType = Binary_Input;
  using UserInterface = InterfaceType;
  // inputs
  std::vector<Binary_Input> add_inputs;
  std::vector<std::string> add_inputs_name;

  std::vector<Binary_Input> sub_inputs;
  std::vector<std::string> sub_inputs_name;

  std::vector<Binary_Input> mul_inputs;
  std::vector<std::string> mul_inputs_name;

  std::vector<Binary_Input> div_inputs;
  std::vector<std::string> div_inputs_name;
  std::map<std::string, int> test_case = {{"add", 0},
                                          {"sub", 1},
                                          {"mul", 2},
                                          {"div", 3}};

};
TYPED_TEST_CASE_P(BinaryOPTest);

TYPED_TEST_P(BinaryOPTest, FourTests) {
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
      switch (test_case_index) {
        case 0:
          aitisa_add(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
          break;
        case 1:
          aitisa_sub(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
          break;
        case 2:
          aitisa_mul(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
          break;
        case 3:
          aitisa_div(aitisa_tensor1, aitisa_tensor2, &aitisa_result);
          break;
        default:
          break;
      }
      aitisa_end = std::clock();
      aitisa_time = (double)(aitisa_end - aitisa_start) / CLOCKS_PER_SEC * 1000;
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
      switch (test_case_index) {
        case 0:
          UserFuncs::user_add(user_tensor1, user_tensor2, &user_result);
          break;
        case 1:
          UserFuncs::user_sub(user_tensor1, user_tensor2, &user_result);
          break;
        case 2:
          UserFuncs::user_mul(user_tensor1, user_tensor2, &user_result);
          break;
        case 3:
          UserFuncs::user_div(user_tensor1, user_tensor2, &user_result);
          break;
        default:
          break;
      }
      user_end = std::clock();
      user_time = (double)(user_end - user_start) / CLOCKS_PER_SEC * 1000;
      UserFuncs::user_resolve(user_result, &user_result_dtype,
                              &user_result_device, &user_result_dims,
                              &user_result_ndim, (void**)&user_result_data,
                              &user_result_len);
      // compare
      int64_t tensor_size = 1;
      ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
      if (test_case_index == 1) {  // CUDA
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
      switch (test_case_index) {
        case 0: {
          auto* aitisa_data = (int32_t*)aitisa_result_data;
          auto* user_data = (int32_t*)user_result_data;
          for (int64_t j = 0; j < tensor_size; j++) {
            ASSERT_EQ(aitisa_data[j], user_data[j]);
          }
          break;
        }
        case 1: {
          auto* aitisa_data = (double*)aitisa_result_data;
          auto* user_data = (double*)user_result_data;
          for (int64_t j = 0; j < tensor_size; j++) {
            ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
          }
          break;
        }
        case 2: {
          auto* aitisa_data = (uint64_t*)aitisa_result_data;
          auto* user_data = (uint64_t*)user_result_data;
          for (int64_t j = 0; j < tensor_size; j++) {
            ASSERT_EQ(aitisa_data[j], user_data[j]);
          }
          break;
        }
        case 3: {
          auto* aitisa_data = (float*)aitisa_result_data;
          auto* user_data = (float*)user_result_data;
          for (int64_t j = 0; j < tensor_size; j++) {
            ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
          }
          break;
        }
        default:
          break;
      }
      // print result of test
      std::cout << /*GREEN <<*/ "[ " << test_case_name << " sample" << i
                << " / " << inputs_name[i] << " ] " << /*RESET <<*/ std::endl;
      std::cout << /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time
                << " ms" << std::endl;
      std::cout << /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time
                << " ms" << std::endl;
    }
  };
  test(std::move(this->add_inputs), std::move(this->add_inputs_name), "add",
       this->test_case["add"]);
  test(std::move(this->sub_inputs), std::move(this->sub_inputs_name), "sub",
       this->test_case["sub"]);
  test(std::move(this->mul_inputs), std::move(this->mul_inputs_name), "mul",
       this->test_case["mul"]);
  test(std::move(this->div_inputs), std::move(this->div_inputs_name), "div",
       this->test_case["div"]);
}
REGISTER_TYPED_TEST_CASE_P(BinaryOPTest, FourTests);

#define REGISTER_BINARY_OP(ADD_FUNC, ADD, SUB_FUNC, SUB, MUL_FUNC, MUL,        \
                           DIV_FUNC, DIV)                                      \
  class BinaryOP : public Basic {                                              \
   public:                                                                     \
    static void user_add(UserTensor tensor1, UserTensor tensor2,               \
                         UserTensor* result) {                                 \
      typedef std::function<void(UserTensor, UserTensor, UserTensor*)>         \
          add_func;                                                            \
      auto func_args_num = aitisa_api::function_traits<ADD_FUNC>::nargs;       \
      auto args_num = aitisa_api::function_traits<add_func>::nargs;            \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<                                                       \
              std::remove_cv<                                                  \
                  aitisa_api::function_traits<add_func>::result_type>::type,   \
              aitisa_api::function_traits<ADD_FUNC>::result_type>::value) {    \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<aitisa_api::function_traits<add_func>::nargs,    \
                              add_func, ADD_FUNC>();                           \
      ADD(tensor1, tensor2, result);                                           \
    }                                                                          \
    static void user_sub(UserTensor tensor1, UserTensor tensor2,               \
                         UserTensor* result) {                                 \
      typedef std::function<void(UserTensor, UserTensor, UserTensor*)>         \
          sub_func;                                                            \
      auto func_args_num = aitisa_api::function_traits<SUB_FUNC>::nargs;       \
      auto args_num = aitisa_api::function_traits<sub_func>::nargs;            \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<                                                       \
              std::remove_cv<                                                  \
                  aitisa_api::function_traits<sub_func>::result_type>::type,   \
              aitisa_api::function_traits<SUB_FUNC>::result_type>::value) {    \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<aitisa_api::function_traits<sub_func>::nargs,    \
                              sub_func, SUB_FUNC>();                           \
      SUB(tensor1, tensor2, result);                                           \
    }                                                                          \
    static void user_mul(UserTensor tensor1, UserTensor tensor2,               \
                         UserTensor* result) {                                 \
      typedef std::function<void(UserTensor, UserTensor, UserTensor*)>         \
          mul_func;                                                            \
      auto func_args_num = aitisa_api::function_traits<MUL_FUNC>::nargs;       \
      auto args_num = aitisa_api::function_traits<mul_func>::nargs;            \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<                                                       \
              std::remove_cv<                                                  \
                  aitisa_api::function_traits<mul_func>::result_type>::type,   \
              aitisa_api::function_traits<MUL_FUNC>::result_type>::value) {    \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<aitisa_api::function_traits<mul_func>::nargs,    \
                              mul_func, MUL_FUNC>();                           \
      MUL(tensor1, tensor2, result);                                           \
    }                                                                          \
    static void user_div(UserTensor tensor1, UserTensor tensor2,               \
                         UserTensor* result) {                                 \
      typedef std::function<void(UserTensor, UserTensor, UserTensor*)>         \
          div_func;                                                            \
      auto func_args_num = aitisa_api::function_traits<DIV_FUNC>::nargs;       \
      auto args_num = aitisa_api::function_traits<div_func>::nargs;            \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<                                                       \
              std::remove_cv<                                                  \
                  aitisa_api::function_traits<div_func>::result_type>::type,   \
              aitisa_api::function_traits<DIV_FUNC>::result_type>::value) {    \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<aitisa_api::function_traits<div_func>::nargs,    \
                              div_func, DIV_FUNC>();                           \
      DIV(tensor1, tensor2, result);                                           \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, BinaryOPTest, BinaryOP);           \
  }

}  // namespace aitisa_api
