#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include "src/math/sqrt.h"
#include "src/nn/relu.h"
#include "src/nn/sigmoid.h"
#include "src/nn/tanh.h"
}

namespace aitisa_api {

template <typename InterfaceType>
class ActivationTest : public ::testing::Test {
 public:
  ActivationTest() {
    fetch_test_data("activate.relu", relu_inputs, relu_inputs_name);
    fetch_test_data("activate.sigmoid", sigmoid_inputs, sigmoid_inputs_name);
    fetch_test_data("activate.tanh", tanh_inputs, tanh_inputs_name);
    fetch_test_data("activate.sqrt", sqrt_inputs, sqrt_inputs_name);
  }
  ~ActivationTest() override = default;

  int fetch_test_data(const char* path, std::vector<Unary_Input>& inputs,
                      std::vector<std::string>& inputs_name) {

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

        std::vector<int64_t> dims;
        int test_index, ndim, dtype, device, len;
        std::string input_name;
        if (!setting.lookupValue("test_index", test_index)) {
          std::cerr << "Setting \"test_index\" do not exist in " << path
                    << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("ndim", ndim)) {
          std::cerr << "Setting \"ndim\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& dims_setting = setting.lookup("dims");
          if (dims_setting.getLength() != ndim) {
            std::cerr << " \"dims\" length is not correct in test index "
                      << test_index << " from " << path << " !\n"
                      << std::endl;
            continue;
          }
          for (int n = 0; n < dims_setting.getLength(); ++n) {
            dims.push_back((int64_t) int(dims_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"dims\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("dtype", dtype)) {
          std::cerr << "Setting \"dtype\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("device", device)) {
          std::cerr << "Setting \"device\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("len", len)) {
          std::cerr << "Setting \"len\" do not exist in test index "
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
        Unary_Input tmp(ndim, dims, dtype, device, nullptr, len);
        inputs.push_back(std::move(tmp));
        inputs_name.push_back(input_name);
      }

      for (auto& input : inputs) {
        unsigned int input_nelem = 1;
        for (unsigned int j = 0; j < input.ndim(); j++) {
          input_nelem *= input.dims()[j];
        }
        unsigned int input_len = input_nelem * elem_size(input.dtype());
        void* input_data = (void*)new char[input_len];
        random_assign(input_data, input_len, input.dtype());
        input.set_data(input_data, input_len);
      }
    } catch (const libconfig::SettingNotFoundException& nfex) {
      std::cerr << nfex.getPath() << " do not exist! " << std::endl;
      return (EXIT_FAILURE);
    }
    return (EXIT_SUCCESS);
  }

  using InputType = Unary_Input;
  using UserInterface = InterfaceType;
  std::vector<Unary_Input> relu_inputs;
  std::vector<std::string> relu_inputs_name;

  std::vector<Unary_Input> sigmoid_inputs;
  std::vector<std::string> sigmoid_inputs_name;

  std::vector<Unary_Input> tanh_inputs;
  std::vector<std::string> tanh_inputs_name;

  std::vector<Unary_Input> sqrt_inputs;
  std::vector<std::string> sqrt_inputs_name;
  std::map<std::string, int> test_case = {{"relu", 0},
                                          {"sigmoid", 1},
                                          {"tanh", 2},
                                          {"sqrt", 3}};
};
TYPED_TEST_CASE_P(ActivationTest);

TYPED_TEST_P(ActivationTest, FourTests) {
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
  auto test = [&m](std::vector<Unary_Input>&& inputs,
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
        AITISA_Tensor aitisa_tensor, aitisa_result;
        AITISA_DataType aitisa_result_dtype;
        AITISA_Device aitisa_result_device;
        UserTensor user_tensor, user_result;
        UserDataType user_result_dtype;
        UserDevice user_result_device;
#ifdef AITISA_API_PYTORCH
        int64_t torch_result_ndim;
        int64_t* torch_result_dims = nullptr;
        float* torch_result_data = nullptr;
        unsigned int torch_result_len;
        TorchTensor torch_tensor, torch_result;
        TorchDataType torch_result_dtype;
        TorchDevice torch_result_device(c10::DeviceType::CPU);
#endif
        // aitisa
        AITISA_DataType aitisa_dtype = aitisa_int_to_dtype(inputs[i].dtype());
        AITISA_Device aitisa_device =
            aitisa_int_to_device(0);  // cpu supoorted only
        aitisa_create(aitisa_dtype, aitisa_device, inputs[i].dims(),
                      inputs[i].ndim(), (void*)(inputs[i].data()),
                      inputs[i].len(), &aitisa_tensor);

        auto aitisa_start = std::chrono::steady_clock::now();
        if (test_case_index == 3) {
          full_float(aitisa_tensor, 1);
        }

        switch (test_case_index) {
          case 0:
            aitisa_relu(aitisa_tensor, &aitisa_result);
            break;
          case 1:
            aitisa_sigmoid(aitisa_tensor, &aitisa_result);
            break;
          case 2:
            aitisa_tanh(aitisa_tensor, &aitisa_result);
            break;
          case 3:
            aitisa_sqrt(aitisa_tensor, &aitisa_result);
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
        UserDataType user_dtype =
            UserFuncs::user_int_to_dtype(inputs[i].dtype());
        UserDevice user_device =
            UserFuncs::user_int_to_device(inputs[i].device());
        UserFuncs::user_create(user_dtype, user_device, inputs[i].dims(),
                               inputs[i].ndim(), inputs[i].data(),
                               inputs[i].len(), &user_tensor);

        auto user_start = std::chrono::steady_clock::now();
        switch (test_case_index) {
          case 0:
            UserFuncs::user_relu(user_tensor, &user_result);
            break;
          case 1:
            UserFuncs::user_sigmoid(user_tensor, &user_result);
            break;
          case 2:
            UserFuncs::user_tanh(user_tensor, &user_result);
            break;
          case 3:
            UserFuncs::user_sqrt(user_tensor, &user_result);
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
        TorchDataType torch_dtype =
            libtorch_api::torch_int_to_dtype(inputs[i].dtype());
        TorchDevice torch_device =
            libtorch_api::torch_int_to_device(inputs[i].device());
        libtorch_api::torch_create(torch_dtype, torch_device, inputs[i].dims(),
                                   inputs[i].ndim(), inputs[i].data(),
                                   inputs[i].len(), &torch_tensor);

        auto torch_start = std::chrono::steady_clock::now();
        switch (test_case_index) {
          case 0:
            torch_result = torch::relu(torch_tensor);
            break;
          case 1:
            torch_result = torch::sigmoid(torch_tensor);
            break;
          case 2:
            torch_result = torch::tanh(torch_tensor);
            break;
          case 3:
            torch_result = torch::sqrt(torch_tensor);
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

        ASSERT_EQ(0, UserFuncs::user_device_to_int(user_result_device));

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
            auto* aitisa_data = (double*)aitisa_result_data;
            auto* user_data = (double*)user_result_data;
#ifdef AITISA_API_PYTORCH
            auto* torch_data = (double*)torch_result_data;
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_FLOAT_EQ(aitisa_data[j], torch_data[j]);
            }
#endif
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_FLOAT_EQ(aitisa_data[j], user_data[j]);
            }
            break;
          }
          default: {
            auto* aitisa_data = (float*)aitisa_result_data;
            auto* user_data = (float*)user_result_data;
#ifdef AITISA_API_PYTORCH
            auto* torch_data = (float*)torch_result_data;
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_FLOAT_EQ(aitisa_data[j], torch_data[j]);
            }
#endif
            for (int64_t j = 0; j < tensor_size; j++) {
              ASSERT_FLOAT_EQ(aitisa_data[j], user_data[j]);
            }
            break;
          }
        }
        aitisa_tensor->storage->data = nullptr;
        aitisa_destroy(&aitisa_tensor);
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
  if (this->relu_inputs.size() && this->sigmoid_inputs.size() &&
      this->tanh_inputs.size() && this->sqrt_inputs.size()) {
    test(std::move(this->relu_inputs), std::move(this->relu_inputs_name),
         "relu", this->test_case["relu"]);
    test(std::move(this->sigmoid_inputs), std::move(this->sigmoid_inputs_name),
         "sigmoid", this->test_case["sigmoid"]);
    test(std::move(this->tanh_inputs), std::move(this->tanh_inputs_name),
         "tanh", this->test_case["tanh"]);
    test(std::move(this->sqrt_inputs), std::move(this->sqrt_inputs_name),
         "sqrt", this->test_case["sqrt"]);
#ifdef AITISA_API_GENERATE_FIGURE
    draw_fig_fun(m, "activate");
#endif
  } else
    FAIL() << "No input test case.";
}

REGISTER_TYPED_TEST_CASE_P(ActivationTest, FourTests);

#define REGISTER_ACTIVATION(RELU_FUNC, RELU, SIGMOID_FUNC, SIGMOID, TANH_FUNC, \
                            TANH, SQRT_FUNC, SQRT)                             \
  class Activation : public Basic {                                            \
   public:                                                                     \
    static void user_relu(UserTensor tensor, UserTensor* result) {             \
      typedef std::function<void(UserTensor, UserTensor*)> relu_func;          \
      auto func_args_num = aitisa_api::function_traits<RELU_FUNC>::nargs;      \
      auto args_num = aitisa_api::function_traits<relu_func>::nargs;           \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<                                                       \
              std::remove_cv<                                                  \
                  aitisa_api::function_traits<relu_func>::result_type>::type,  \
              aitisa_api::function_traits<RELU_FUNC>::result_type>::value) {   \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<aitisa_api::function_traits<relu_func>::nargs,   \
                              relu_func, RELU_FUNC>();                         \
      RELU(tensor, result);                                                    \
    }                                                                          \
    static void user_sigmoid(UserTensor tensor, UserTensor* result) {          \
      typedef std::function<void(UserTensor, UserTensor*)> sigmoid_func;       \
      auto func_args_num = aitisa_api::function_traits<SIGMOID_FUNC>::nargs;   \
      auto args_num = aitisa_api::function_traits<sigmoid_func>::nargs;        \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<std::remove_cv<aitisa_api::function_traits<            \
                            sigmoid_func>::result_type>::type,                 \
                        aitisa_api::function_traits<                           \
                            SIGMOID_FUNC>::result_type>::value) {              \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<                                                 \
          aitisa_api::function_traits<sigmoid_func>::nargs, sigmoid_func,      \
          SIGMOID_FUNC>();                                                     \
      SIGMOID(tensor, result);                                                 \
    }                                                                          \
    static void user_tanh(UserTensor tensor, UserTensor* result) {             \
      typedef std::function<void(UserTensor, UserTensor*)> tanh_func;          \
      auto func_args_num = aitisa_api::function_traits<TANH_FUNC>::nargs;      \
      auto args_num = aitisa_api::function_traits<tanh_func>::nargs;           \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<                                                       \
              std::remove_cv<                                                  \
                  aitisa_api::function_traits<tanh_func>::result_type>::type,  \
              aitisa_api::function_traits<TANH_FUNC>::result_type>::value) {   \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<aitisa_api::function_traits<tanh_func>::nargs,   \
                              tanh_func, TANH_FUNC>();                         \
      TANH(tensor, result);                                                    \
    }                                                                          \
    static void user_sqrt(UserTensor tensor, UserTensor* result) {             \
      typedef std::function<void(UserTensor, UserTensor*)> sqrt_func;          \
      auto func_args_num = aitisa_api::function_traits<SQRT_FUNC>::nargs;      \
      auto args_num = aitisa_api::function_traits<sqrt_func>::nargs;           \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<                                                       \
              std::remove_cv<                                                  \
                  aitisa_api::function_traits<sqrt_func>::result_type>::type,  \
              aitisa_api::function_traits<SQRT_FUNC>::result_type>::value) {   \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<aitisa_api::function_traits<sqrt_func>::nargs,   \
                              sqrt_func, SQRT_FUNC>();                         \
      SQRT(tensor, result);                                                    \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, ActivationTest, Activation);       \
  }

}  // namespace aitisa_api
