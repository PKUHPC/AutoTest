#pragma once

#include <chrono>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"

extern "C" {
#include <math.h>
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
        config_setting_t* dims_setting = config_setting_lookup(test, "dims");
        if (!dims_setting) {
          fprintf(stderr, "No 'dims' in test case %d from %s.\n", i, path);
          continue;
        }
        int64_t ndim;
        std::vector<int64_t> dims;
        int dtype, device, len, axis;
        const char* input_name;

        if (!config_setting_lookup_int64(
                test, "ndim", reinterpret_cast<long long int*>(&ndim))) {
          fprintf(stderr, "No 'ndim' in test case %d from %s.\n", i, path);
          continue;
        }
        if (config_setting_length(dims_setting) != ndim) {
          fprintf(stderr,
                  "'dims' length is not correct in test case %d from %s.\n", i,
                  path);
          continue;
        }
        for (int j = 0; j < ndim; ++j) {
          int64_t input = config_setting_get_int_elem(dims_setting, j);
          dims.push_back(input);
        }
        if (!config_setting_lookup_int(test, "dtype", &dtype)) {
          fprintf(stderr, "No 'dtype' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_int(test, "device", &device)) {
          fprintf(stderr, "No 'device' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_int(test, "len", &len)) {
          fprintf(stderr, "No 'len' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_string(test, "input_name", &input_name)) {
          fprintf(stderr, "No 'input_name' in test case %d from %s.\n", i,
                  path);
          continue;
        }
        if (!config_setting_lookup_int(test, "axis", &axis)) {
          fprintf(stderr, "No 'axis' in test case %d from %s.\n", i, path);
          continue;
        }

        Softmax_Input tmp(
            /*ndim*/ ndim, /*dims*/ dims, /*dtype=double*/ dtype,
            /*device=cpu*/ device, /*data*/ nullptr, /*len*/ len, axis);

        inputs.push_back(std::move(tmp));
        inputs_name.emplace_back(input_name);
      }
    } else {
      fprintf(stderr, "Can not find path %s in config.\n", path);
      return (EXIT_FAILURE);
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
    config_destroy(&cfg);
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
      std::chrono::duration<double> aitisa_elapsed = aitisa_end - aitisa_start;
      aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device,
                     &aitisa_result_dims, &aitisa_result_ndim,
                     (void**)&aitisa_result_data, &aitisa_result_len);

      // user
      UserDataType user_dtype = UserFuncs::user_int_to_dtype(inputs[i].dtype());
      UserDevice user_device =
          UserFuncs::user_int_to_device(inputs[i].device());
      UserFuncs::user_create(user_dtype, user_device, inputs[i].dims(),
                             inputs[i].ndim(), inputs[i].data(),
                             inputs[i].len(), &user_tensor);
      auto user_start = std::chrono::steady_clock::now();

      UserFuncs::user_softmax(user_tensor, inputs[i].axis(), &user_result);

      auto user_end = std::chrono::steady_clock::now();
      std::chrono::duration<double> user_elapsed = user_end - user_start;
      UserFuncs::user_resolve(user_result, &user_result_dtype,
                              &user_result_device, &user_result_dims,
                              &user_result_ndim, (void**)&user_result_data,
                              &user_result_len);

      // compare
      int64_t tensor_size = 1;
      ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
      ASSERT_EQ(/*CUDA*/ 0, UserFuncs::user_device_to_int(user_result_device));
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
      //            // print result of test
      std::cout << /*GREEN <<*/ "[ " << test_case_name << " sample" << i
                << " / " << inputs_name[i] << " ] " << /*RESET <<*/ std::endl;
      std::cout << /*GREEN <<*/ "\t[ AITISA ] "
                << /*RESET <<*/ aitisa_elapsed.count() * 1000 << " ms"
                << std::endl;
      std::cout << /*GREEN <<*/ "\t[  USER  ] "
                << /*RESET <<*/ user_elapsed.count() * 1000 << " ms"
                << std::endl;
      m.insert(std::make_pair(test_case_name + " sample " + std::to_string(i),
                              time_map_value(aitisa_elapsed.count() * 1000,
                                             user_elapsed.count() * 1000)));
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
