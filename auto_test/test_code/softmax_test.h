#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"

extern "C" {
#include "src/nn/softmax.h"
}

namespace aitisa_api {

namespace {

class Softmax_Input : public Unary_Input {
 public:
  Softmax_Input() = default;
  Softmax_Input(int64_t ndim, std::vector<int64_t> dims, int dtype, int device,
                void* data, unsigned int len, int axis)
      : Unary_Input(ndim, dims, dtype, device, data, len), axis_(axis) {}
  Softmax_Input(Softmax_Input&& input) noexcept
      : Unary_Input(input.ndim(), input.dims(), input.dtype(), input.device(),
                    input.data(), input.len()),
        axis_(input.axis()) {
    input.to_nullptr();
  }
  ~Softmax_Input() override = default;

  Softmax_Input& operator=(Softmax_Input& right) {
    Unary_Input& left = (Unary_Input&)(*this);
    left = (Unary_Input&)right;
    this->axis_ = right.axis();
  }
  int axis() const { return axis_; }

 private:
  int axis_ = -1;
};
}  // namespace

template <typename InterfaceType>
class SoftmaxTest : public ::testing::Test {
 public:
  SoftmaxTest() { fetch_test_data("softmax", softmax_inputs, softmax_name); }
  ~SoftmaxTest() override = default;
  using InputType = Softmax_Input;
  using UserInterface = InterfaceType;
  static void aitisa_kernel(const AITISA_Tensor input, const int axis,
                            AITISA_Tensor* output) {
    aitisa_softmax(input, axis, output);
  }
  int fetch_test_data(const char* path, std::vector<Softmax_Input>& inputs,
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
        int test_index, ndim, dtype, device, len, axis;
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
        if (!setting.lookupValue("axis", axis)) {
          std::cerr << "Setting \"axis\" do not exist in test index "
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
        Softmax_Input tmp(ndim, dims, dtype, device, nullptr, len, axis);
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
  // inputs
  std::vector<Softmax_Input> softmax_inputs;
  std::vector<std::string> softmax_name;
  std::map<std::string, int> test_case = {{"softmax", 0}};
};
TYPED_TEST_CASE_P(SoftmaxTest);

TYPED_TEST_P(SoftmaxTest, TwoTests) {
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;

  time_map m;
  auto test = [&m](std::vector<Softmax_Input>&& inputs,
                   std::vector<std::string>&& inputs_name,
                   const std::string& test_case_name, int test_case_index) {
    for (int i = 0; i < inputs.size(); i++) {
      auto aitisa_elapsed = std::chrono::duration<double>::zero();
      auto user_elapsed = std::chrono::duration<double>::zero();
      //loop test
      for (int n = 0; n < loop; n++) {
        int64_t aitisa_result_ndim, user_result_ndim;
        int64_t *aitisa_result_dims = nullptr, *user_result_dims = nullptr;
        float *aitisa_result_data = nullptr, *user_result_data = nullptr;
        unsigned int aitisa_result_len, user_result_len;
        AITISA_Tensor aitisa_tensor, aitisa_result;
        AITISA_DataType aitisa_result_dtype;
        AITISA_Device aitisa_result_device;
        UserTensor user_tensor, user_result;
        UserDataType user_result_dtype;
        UserDevice user_result_device;
        // aitisa
        AITISA_DataType aitisa_dtype = aitisa_int_to_dtype(inputs[i].dtype());
        AITISA_Device aitisa_device =
            aitisa_int_to_device(0);  // cpu supoorted only
        aitisa_create(aitisa_dtype, aitisa_device, inputs[i].dims(),
                      inputs[i].ndim(), (void*)(inputs[i].data()),
                      inputs[i].len(), &aitisa_tensor);
        auto aitisa_start = std::chrono::steady_clock::now();

        aitisa_softmax(aitisa_tensor, inputs[i].axis(), &aitisa_result);

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

        UserFuncs::user_softmax(user_tensor, inputs[i].axis(), &user_result);

        auto user_end = std::chrono::steady_clock::now();
        user_elapsed += user_end - user_start;
        UserFuncs::user_resolve(user_result, &user_result_dtype,
                                &user_result_device, &user_result_dims,
                                &user_result_ndim, (void**)&user_result_data,
                                &user_result_len);

        // compare
        int64_t tensor_size = 1;
        ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
        ASSERT_EQ(/*CUDA*/ 0,
                  UserFuncs::user_device_to_int(user_result_device));
        ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype),
                  UserFuncs::user_dtype_to_int(user_result_dtype));
        for (int64_t j = 0; j < aitisa_result_ndim; j++) {
          tensor_size *= aitisa_result_dims[j];
          ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
        }
        ASSERT_EQ(aitisa_result_len, user_result_len);
        auto* aitisa_data = (float*)aitisa_result_data;
        auto* user_data = (float*)user_result_data;
        for (int64_t j = 0; j < tensor_size; j++) {
          ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
        }
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
      m.insert(std::make_pair(test_case_name + " sample " + std::to_string(i),
                              time_map_value(aitisa_time, user_time)));
    }
  };
  if (this->softmax_inputs.size()) {
    test(std::move(this->softmax_inputs), std::move(this->softmax_name),
         "softmax", this->test_case["softmax"]);
#ifdef AITISA_API_GENERATE_FIGURE
    draw_fig_fun(m, "softmax");
#endif
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(SoftmaxTest, TwoTests);

#define REGISTER_SOFTMAX(SOFTMAX_FUNC, SOFTMAX)                                \
  class Softmax : public Basic {                                               \
   public:                                                                     \
    static void user_softmax(UserTensor input, const int axis,                 \
                             UserTensor* output) {                             \
      typedef std::function<void(const UserTensor, const int, UserTensor*)>    \
          softmax_func;                                                        \
      auto func_args_num = aitisa_api::function_traits<SOFTMAX_FUNC>::nargs;   \
      auto args_num = aitisa_api::function_traits<softmax_func>::nargs;        \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<std::remove_cv<aitisa_api::function_traits<            \
                            softmax_func>::result_type>::type,                 \
                        aitisa_api::function_traits<                           \
                            SOFTMAX_FUNC>::result_type>::value) {              \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<                                                 \
          aitisa_api::function_traits<softmax_func>::nargs, softmax_func,      \
          SOFTMAX_FUNC>();                                                     \
      SOFTMAX(input, axis, output);                                            \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, SoftmaxTest, Softmax);             \
  }

}  // namespace aitisa_api
