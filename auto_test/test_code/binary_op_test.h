#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"

extern "C" {
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
  ~BinaryOPTest() override = default;
  int fetch_test_data(const char* path, std::vector<Binary_Input>& inputs,
                      std::vector<std::string>& inputs_name,
                      int test_case_index) {

    libconfig::Config cfg;

    try {
      cfg.readFile(CONFIG_FILE);
    } catch (const libconfig::FileIOException& fioex) {
      std::cerr << "I/O error while reading file." << std::endl;
      return (EXIT_FAILURE);
    } catch (const libconfig::ParseException& pex) {
      std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                << " - " << pex.getError() << std::endl;
      return (EXIT_FAILURE);
    }

    try {
      const libconfig::Setting& settings = cfg.lookup(path);
      int count = settings.getLength();

      for (int i = 0; i < count; ++i) {
        const libconfig::Setting& setting = settings[i];

        std::vector<int64_t> dims1, dims2;
        int dtype1, device1, len1, dtype2, device2, len2, ndim1, ndim2,
            test_index;
        std::string input_name;

        if (!setting.lookupValue("test_index", test_index)) {
          std::cerr << "Setting \"test_index\" do not exist in " << path
                    << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("ndim1", ndim1)) {
          std::cerr << "Setting \"ndim1\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& dims1_setting = setting.lookup("dims1");
          if (dims1_setting.getLength() != ndim1) {
            std::cerr << " \"dims1\" length is not correct in test index "
                      << test_index << " from " << path << " !\n"
                      << std::endl;
            continue;
          }
          for (int n = 0; n < dims1_setting.getLength(); ++n) {
            dims1.push_back((int64_t) int(dims1_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"dims1\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("ndim2", ndim2)) {
          std::cerr << "Setting \"ndim2\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& dims2_setting = setting.lookup("dims2");
          if (dims2_setting.getLength() != ndim2) {
            std::cerr << " \"dims2\" length is not correct in test index "
                      << test_index << " from " << path << " !\n"
                      << std::endl;
            continue;
          }
          for (int n = 0; n < dims2_setting.getLength(); ++n) {
            dims2.push_back((int64_t) int(dims2_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"dims2\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("dtype1", dtype1)) {
          std::cerr << "Setting \"dtype1\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("device1", device1)) {
          std::cerr << "Setting \"device1\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("len1", len1)) {
          std::cerr << "Setting \"len1\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("dtype2", dtype2)) {
          std::cerr << "Setting \"dtype2\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("device2", device2)) {
          std::cerr << "Setting \"device2\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("len2", len2)) {
          std::cerr << "Setting \"len2\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("input_name", input_name)) {
          std::cerr << "Setting \"input_name\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        Binary_Input tmp(ndim1, dims1, dtype1, device1, nullptr, len1, ndim2,
                         dims2, dtype2, device2, nullptr, len2);
        inputs.push_back(std::move(tmp));
        inputs_name.push_back(input_name);
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

    } catch (const libconfig::SettingNotFoundException& nfex) {
      std::cerr << nfex.getPath() << " do not exist! " << std::endl;
      return (EXIT_FAILURE);
    }
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
#ifdef AITISA_API_PYTORCH
  using TorchDataType = typename libtorch_api::DataType;
  using TorchDevice = typename libtorch_api::Device;
  using TorchTensor = typename libtorch_api::Tensor;
#endif
  time_map m;
  auto test = [&m](std::vector<Binary_Input>&& inputs,
                   std::vector<std::string>&& inputs_name,
                   const std::string& test_case_name, int test_case_index) {
    for (int i = 0; i < inputs.size(); i++) {
      auto aitisa_elapsed = std::chrono::duration<double>::zero();
      auto user_elapsed = std::chrono::duration<double>::zero();
#ifdef AITISA_API_PYTORCH
      auto torch_elapsed = std::chrono::duration<double>::zero();
#endif
      //loop test
      for (int n = 0; n < loop; n++) {
        int64_t aitisa_result_ndim, user_result_ndim;
        int64_t *aitisa_result_dims = nullptr, *user_result_dims = nullptr;
        void *aitisa_result_data = nullptr, *user_result_data = nullptr;
        unsigned int aitisa_result_len, user_result_len;
        AITISA_Tensor aitisa_tensor1, aitisa_tensor2, aitisa_result;
        AITISA_DataType aitisa_result_dtype;
        AITISA_Device aitisa_result_device;
        UserTensor user_tensor1, user_tensor2, user_result;
        UserDataType user_result_dtype;
        UserDevice user_result_device;
#ifdef AITISA_API_PYTORCH
        int64_t torch_result_ndim;
        int64_t* torch_result_dims = nullptr;
        void* torch_result_data = nullptr;
        unsigned int torch_result_len;
        TorchTensor torch_tensor1, torch_tensor2, torch_result;
        TorchDataType torch_result_dtype;
        TorchDevice torch_result_device(c10::DeviceType::CPU);
#endif
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
        auto aitisa_start = std::chrono::steady_clock::now();
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
        auto aitisa_end = std::chrono::steady_clock::now();
        aitisa_elapsed += aitisa_end - aitisa_start;
        aitisa_resolve(aitisa_result, &aitisa_result_dtype,
                       &aitisa_result_device, &aitisa_result_dims,
                       &aitisa_result_ndim, (void**)&aitisa_result_data,
                       &aitisa_result_len);
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
        auto user_start = std::chrono::steady_clock::now();
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
        auto user_end = std::chrono::steady_clock::now();
        user_elapsed += user_end - user_start;

        UserFuncs::user_resolve(user_result, &user_result_dtype,
                                &user_result_device, &user_result_dims,
                                &user_result_ndim, (void**)&user_result_data,
                                &user_result_len);
#ifdef AITISA_API_PYTORCH
        //torch
        TorchDataType torch_dtype1 =
            libtorch_api::torch_int_to_dtype(inputs[i].dtype1());
        TorchDataType torch_dtype2 =
            libtorch_api::torch_int_to_dtype(inputs[i].dtype2());
        TorchDevice torch_device1 =
            libtorch_api::torch_int_to_device(inputs[i].device1());
        TorchDevice torch_device2 =
            libtorch_api::torch_int_to_device(inputs[i].device2());
        libtorch_api::torch_create(
            torch_dtype1, torch_device1, inputs[i].dims1(), inputs[i].ndim1(),
            inputs[i].data1(), inputs[i].len1(), &torch_tensor1);
        libtorch_api::torch_create(
            torch_dtype2, torch_device2, inputs[i].dims2(), inputs[i].ndim2(),
            inputs[i].data2(), inputs[i].len2(), &torch_tensor2);

        auto torch_start = std::chrono::steady_clock::now();
        switch (test_case_index) {
          case 0:
            torch_result = torch::add(torch_tensor1, torch_tensor2);
            break;
          case 1:
            torch_result = torch::sub(torch_tensor1, torch_tensor2);
            break;
          case 2:
            torch_result = torch::mul(torch_tensor1, torch_tensor2);
            break;
          case 3:
            torch_result = torch::div(torch_tensor1, torch_tensor2);
            break;
          default:
            break;
        }
        auto torch_end = std::chrono::steady_clock::now();
        torch_elapsed += torch_end - torch_start;
        libtorch_api::torch_resolve(
            torch_result, &torch_result_dtype, torch_result_device,
            &torch_result_dims, &torch_result_ndim, (void**)&torch_result_data,
            &torch_result_len);
#endif

        // compare
        int64_t tensor_size = 1;
        ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
        if (test_case_index == 1) {
          ASSERT_EQ(0, UserFuncs::user_device_to_int(user_result_device));
        } else {
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
#ifdef AITISA_API_PYTORCH
        ASSERT_EQ(aitisa_result_ndim, torch_result_ndim);
        ASSERT_EQ(0, libtorch_api::torch_device_to_int(torch_result_device));
        ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype),
                  libtorch_api::torch_dtype_to_int(torch_result_dtype));
        for (int64_t j = 0; j < aitisa_result_ndim; j++) {
          ASSERT_EQ(aitisa_result_dims[j], torch_result_dims[j]);
        }
        ASSERT_EQ(aitisa_result_len, torch_result_len);
#endif
        switch (test_case_index) {
          case 0: {
            auto* aitisa_data = (int32_t*)aitisa_result_data;
            auto* user_data = (int32_t*)user_result_data;
#ifdef AITISA_API_PYTORCH
            auto* torch_data = (int32_t*)torch_result_data;
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_EQ(aitisa_data[j], torch_data[j]);
            }
#endif
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_EQ(aitisa_data[j], user_data[j]);
            }
            break;
          }
          case 1: {
            auto* aitisa_data = (double*)aitisa_result_data;
            auto* user_data = (double*)user_result_data;
#ifdef AITISA_API_PYTORCH
            auto* torch_data = (double*)torch_result_data;
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_TRUE(abs(aitisa_data[j] - torch_data[j]) < 1e-3);
            }
#endif
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
            }
            break;
          }
          case 2: {
            auto* aitisa_data = (float*)aitisa_result_data;
            auto* user_data = (float*)user_result_data;
#ifdef AITISA_API_PYTORCH
            auto* torch_data = (float*)torch_result_data;
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_EQ(aitisa_data[j], torch_data[j]);
            }
#endif
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_EQ(aitisa_data[j], user_data[j]);
            }
            break;
          }
          case 3: {
            auto* aitisa_data = (float*)aitisa_result_data;
            auto* user_data = (float*)user_result_data;
#ifdef AITISA_API_PYTORCH
            auto* torch_data = (float*)torch_result_data;
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_TRUE(abs(aitisa_data[j] - torch_data[j]) < 1e-3);
            }
#endif
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
            }
            break;
          }
          default:
            break;
        }
        aitisa_tensor1->storage->data = nullptr;
        aitisa_tensor2->storage->data = nullptr;
        aitisa_destroy(&aitisa_tensor1);
        aitisa_destroy(&aitisa_tensor2);
        aitisa_destroy(&aitisa_result);
      }
      auto aitisa_time = aitisa_elapsed.count() * 1000 / loop;
      auto user_time = user_elapsed.count() * 1000 / loop;

      // print result of test
      std::cout << "[ " << test_case_name << " sample" << i << " / "
                << inputs_name[i] << " ] " << std::endl;
      std::cout << "\t[ AITISA ] " << aitisa_time << " ms average for " << loop
                << " loop " << std::endl;
      std::cout << "\t[  USER  ] " << user_time << " ms average for " << loop
                << " loop " << std::endl;
#ifdef AITISA_API_PYTORCH
      auto torch_time = torch_elapsed.count() * 1000 / loop;
      std::cout << "\t[  TORCH  ] " << torch_time << " ms average for " << loop
                << " loop " << std::endl;
      m.insert(
          std::make_pair(test_case_name + " sample " + std::to_string(i),
                         time_map_value(aitisa_time, user_time, torch_time)));
#else
      m.insert(std::make_pair(test_case_name + " sample " + std::to_string(i),
                              time_map_value(aitisa_time, user_time)));
#endif
    }
  };
  if (this->add_inputs.size() && this->sub_inputs.size() &&
      this->mul_inputs.size() && this->div_inputs.size()) {
    test(std::move(this->add_inputs), std::move(this->add_inputs_name), "add",
         this->test_case["add"]);
    test(std::move(this->sub_inputs), std::move(this->sub_inputs_name), "sub",
         this->test_case["sub"]);
    test(std::move(this->mul_inputs), std::move(this->mul_inputs_name), "mul",
         this->test_case["mul"]);
    test(std::move(this->div_inputs), std::move(this->div_inputs_name), "div",
         this->test_case["div"]);
#ifdef AITISA_API_GENERATE_FIGURE
    draw_fig_fun(m, "binary_op");
#endif
  } else
    FAIL() << "No input test case.";
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
