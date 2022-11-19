#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
#include "hice/core/tensor_printer.h"

namespace aitisa_api {

namespace {

class ArgReduce_Input : public Unary_Input {
 public:
  ArgReduce_Input() = default;

  ArgReduce_Input(int64_t ndim, std::vector<int64_t> dims, int dtype,
                  int device, void* data, unsigned int len, int64_t dim)
      : Unary_Input(ndim, std::move(dims), dtype, device, data, len),
        dim_(dim) {}
  ArgReduce_Input(ArgReduce_Input&& input) noexcept
      : Unary_Input(input.ndim(), input.dims(), input.dtype(), input.device(),
                    input.data(), input.len()),
        dim_(input.dim()) {
    input.to_nullptr();
  }
  ~ArgReduce_Input() override = default;

  ArgReduce_Input& operator=(ArgReduce_Input& right) {
    auto& left = (Unary_Input&)(*this);
    left = (Unary_Input&)right;
    this->dim_ = right.dim();
  }
  float dim() const { return dim_; }

 private:
  int64_t dim_ = 0;
};
}  // namespace

template <typename InterfaceType>
class ArgReduceTest : public ::testing::Test {
 public:
  ArgReduceTest() {
    fetch_test_data("arg_reduce.argmin", argmin_inputs, argmin_inputs_name);
    fetch_test_data("arg_reduce.argmax", argmax_inputs, argmax_inputs_name);
  }
  ~ArgReduceTest() override = default;

  int fetch_test_data(const char* path, std::vector<ArgReduce_Input>& inputs,
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
        int test_index, ndim, dtype, device, len, dim;
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
        if (!setting.lookupValue("dim", dim)) {
          std::cerr << "Setting \"dim\" do not exist in test index "
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
        ArgReduce_Input tmp(ndim, dims, dtype, device, nullptr, len, dim);
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
  using InputType = ArgReduce_Input;
  using UserInterface = InterfaceType;
  // inputs
  std::vector<ArgReduce_Input> argmin_inputs;
  std::vector<std::string> argmin_inputs_name;

  std::vector<ArgReduce_Input> argmax_inputs;
  std::vector<std::string> argmax_inputs_name;
  std::map<std::string, int> test_case = {{"argmin", 0}, {"argmax", 1}};
};
TYPED_TEST_CASE_P(ArgReduceTest);

TYPED_TEST_P(ArgReduceTest, TwoTests) {
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
  auto test = [&m](std::vector<ArgReduce_Input>&& inputs,
                   std::vector<std::string>&& inputs_name,
                   const std::string& test_case_name, int test_case_index) {
    for (int i = 0; i < inputs.size(); i++) {
      auto user_elapsed = std::chrono::duration<double>::zero();
#ifdef AITISA_API_PYTORCH
      auto torch_elapsed = std::chrono::duration<double>::zero();
#endif
      //loop test
      for (int n = 0; n < loop; n++) {
        int64_t user_result_ndim;
        int64_t* user_result_dims = nullptr;
        void* user_result_data = nullptr;
        unsigned int user_result_len;
        UserTensor user_tensor, user_result;
        UserDataType user_result_dtype;
        UserDevice user_result_device;
#ifdef AITISA_API_PYTORCH
        int64_t torch_result_ndim;
        int64_t* torch_result_dims = nullptr;
        void* torch_result_data = nullptr;
        unsigned int torch_result_len;
        TorchTensor torch_tensor, torch_result;
        TorchDataType torch_result_dtype;
        TorchDevice torch_result_device(c10::DeviceType::CPU);
#endif
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
            user_result = UserFuncs::user_argmin(user_tensor, inputs[i].dim());
            break;
          case 1:
            user_result = UserFuncs::user_argmin(user_tensor, inputs[i].dim());
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
            torch_result = torch::argmin(torch_tensor, inputs[i].dim());
            break;
          case 1:
            torch_result = torch::argmax(torch_tensor, inputs[i].dim());
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
#ifdef AITISA_API_PYTORCH
        ASSERT_EQ(user_result_ndim, torch_result_ndim);
        int64_t tensor_size = 1;
        ASSERT_EQ(UserFuncs::user_device_to_int(user_result_device),
                  libtorch_api::torch_device_to_int(torch_result_device));
        ASSERT_EQ(UserFuncs::user_dtype_to_int(user_result_dtype),
                  libtorch_api::torch_dtype_to_int(torch_result_dtype));
        for (int64_t j = 0; j < user_result_ndim; j++) {
          tensor_size *= user_result_dims[j];
          ASSERT_EQ(user_result_dims[j], torch_result_dims[j]);
        }
        ASSERT_EQ(user_result_len, torch_result_len);
#endif
        auto* user_data = (float*)user_result_data;
        auto* torch_data = (float*)torch_result_data;
        for (int64_t j = 0; j < tensor_size; j++) {
          ASSERT_TRUE(abs(user_data[j] - torch_data[j]) < 1e-3);
        }
      }
      auto user_time = user_elapsed.count() * 1000 / loop;

      // print result of test
      std::cout << "[ " << test_case_name << " sample" << i << " / "
                << inputs_name[i] << " ] " << std::endl;
      std::cout << "\t[  USER  ] " << user_time << " ms average for " << loop
                << " loop " << std::endl;
#ifdef AITISA_API_PYTORCH
      auto torch_time = torch_elapsed.count() * 1000 / loop;
      std::cout << "\t[  TORCH  ] " << torch_time << " ms average for " << loop
                << " loop " << std::endl;
//      m.insert(
//          std::make_pair(test_case_name + " sample " + std::to_string(i),
//                         time_map_value(aitisa_time, user_time, torch_time)));
#else
//      m.insert(std::make_pair(test_case_name + " sample " + std::to_string(i),
//                              time_map_value(aitisa_time, user_time)));
#endif
    }
  };
  if (this->argmin_inputs.size() && this->argmax_inputs.size()) {
    test(std::move(this->argmin_inputs), std::move(this->argmin_inputs_name),
         "argmin", this->test_case["argmin"]);
    test(std::move(this->argmax_inputs), std::move(this->argmax_inputs_name),
         "argmax", this->test_case["argmax"]);
#ifdef AITISA_API_GENERATE_FIGURE
//    draw_fig_fun(m, "drop_out");
#endif
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(ArgReduceTest, TwoTests);

#define REGISTER_ARGREDUCE(ARGMIN, ARGMAX)                               \
  class ArgReduce : public Basic {                                       \
   public:                                                               \
    static UserTensor user_argmin(UserTensor input, const int64_t dim) { \
      return ARGMIN(input, dim, false);                                  \
    }                                                                    \
    static UserTensor user_argmax(UserTensor input, const int64_t dim) { \
      return ARGMAX(input, dim, false);                                  \
    }                                                                    \
  };                                                                     \
  namespace aitisa_api {                                                 \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, ArgReduceTest, ArgReduce);   \
  }

}  // namespace aitisa_api
