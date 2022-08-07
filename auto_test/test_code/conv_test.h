#pragma once

#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include <math.h>
#include <sys/time.h>
#include "src/nn/conv.h"
}

namespace aitisa_api {

namespace {

class Conv_Input : public Binary_Input {
 public:
  Conv_Input() = default;
  ;
  Conv_Input(int64_t ndim1, int64_t* dims1, int dtype1, int device1,
             void* data1, unsigned int len1, int64_t ndim2, int64_t* dims2,
             int dtype2, int device2, void* data2, unsigned int len2,
             int* stride, int* padding, int* dilation, int groups)
      : Binary_Input(ndim1, dims1, dtype1, device1, data1, len1, ndim2, dims2,
                     dtype2, device2, data2, len2),
        stride_(stride),
        padding_(padding),
        dilation_(dilation),
        groups_(groups) {}
  Conv_Input(int64_t ndim1, std::vector<int64_t> dims1, int dtype1, int device1,
             void* data1, unsigned int len1, int64_t ndim2,
             std::vector<int64_t> dims2, int dtype2, int device2, void* data2,
             unsigned int len2, std::vector<int> stride,
             std::vector<int> padding, std::vector<int> dilation, int groups)
      : Binary_Input(ndim1, std::move(dims1), dtype1, device1, data1, len1,
                     ndim2, std::move(dims2), dtype2, device2, data2, len2),
        stride_(nullptr),
        padding_(nullptr),
        dilation_(nullptr),
        groups_(groups) {
    int spatial_len = ndim1 - 2;
    this->stride_ = new int[spatial_len];
    this->padding_ = new int[spatial_len];
    this->dilation_ = new int[spatial_len];
    for (int i = 0; i < spatial_len; i++) {
      this->stride_[i] = stride[i];
      this->padding_[i] = padding[i];
      this->dilation_[i] = dilation[i];
    }
  }
  Conv_Input(Conv_Input&& input) noexcept
      : Binary_Input(input.ndim1(), input.dims1(), input.dtype1(),
                     input.device1(), input.data1(), input.len1(),
                     input.ndim2(), input.dims2(), input.dtype2(),
                     input.device2(), input.data2(), input.len2()),
        stride_(input.stride()),
        padding_(input.padding()),
        dilation_(input.dilation()),
        groups_(input.groups()) {
    input.to_nullptr();
    input.stride_ = nullptr;
    input.padding_ = nullptr;
    input.dilation_ = nullptr;
  }
  ~Conv_Input() override {
    delete[] stride_;
    delete[] padding_;
    delete[] dilation_;
  }
  Conv_Input& operator=(Conv_Input& right) {
    int spatial_len = right.ndim1() - 2;
    auto& left = (Binary_Input&)(*this);
    left = (Binary_Input&)right;
    this->stride_ = new int[spatial_len];
    this->padding_ = new int[spatial_len];
    this->dilation_ = new int[spatial_len];
    memcpy(this->stride_, right.stride(), spatial_len * sizeof(int));
    memcpy(this->padding_, right.padding(), spatial_len * sizeof(int));
    memcpy(this->dilation_, right.dilation(), spatial_len * sizeof(int));
  }
  int* stride() { return stride_; }
  int* padding() { return padding_; }
  int* dilation() { return dilation_; }
  int groups() const { return groups_; }

 private:
  int* stride_ = nullptr;
  int* padding_ = nullptr;
  int* dilation_ = nullptr;
  int groups_ = 1;
};

}  // namespace

template <typename InterfaceType>
class Conv2dTest : public ::testing::Test {
 public:
  Conv2dTest() { fetch_test_data("conv2d", conv2d_inputs, conv2d_inputs_name); }
  ~Conv2dTest() override = default;

  static void aitisa_kernel(const AITISA_Tensor input,
                            const AITISA_Tensor filter, int* stride,
                            const int* padding, const int* dilation,
                            const int groups, AITISA_Tensor* output) {
    aitisa_conv2d(input, filter, stride, 2, padding, 2, dilation, 2, groups,
                  output);
  }

  int fetch_test_data(const char* path, std::vector<Conv_Input>& inputs,
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
        if (!dims1_setting) {
          fprintf(stderr, "No 'dims1' in test case %d from %s.\n", i, path);
          continue;
        }
        config_setting_t* dims2_setting = config_setting_lookup(test, "dims2");
        if (!dims2_setting) {
          fprintf(stderr, "No 'dims2' in test case %d from %s.\n", i, path);
          continue;
        }

        int64_t ndim1, ndim2;
        std::vector<int64_t> dims1, dims2;
        int dtype1, device1, len1, dtype2, device2, len2;
        const char* input_name;
        std::vector<int> stride, padding, dilation;
        int groups;

        if (!config_setting_lookup_int64(
                test, "ndim1", reinterpret_cast<long long int*>(&ndim1))) {
          fprintf(stderr, "No 'ndim1' in test case %d from %s.\n", i, path);
          continue;
        }
        if (config_setting_length(dims1_setting) != ndim1) {
          fprintf(stderr,
                  "'dims1' length is not correct in test case %d from %s.\n", i,
                  path);
          continue;
        }
        if (!config_setting_lookup_int64(
                test, "ndim2", reinterpret_cast<long long int*>(&ndim2))) {
          fprintf(stderr, "No 'ndim2' in test case %d from %s.\n", i, path);
          continue;
        }
        if (config_setting_length(dims2_setting) != ndim2) {
          fprintf(stderr,
                  "'dims2' length is not correct in test case %d from %s.\n", i,
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
        if (!config_setting_lookup_int(test, "groups", &groups)) {
          fprintf(stderr, "No 'groups' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_string(test, "input_name", &input_name)) {
          fprintf(stderr, "No 'input_name' in test case %d from %s.\n", i,
                  path);
          continue;
        }
        config_setting_t* stride_setting =
            config_setting_get_member(test, "stride");
        if (!stride_setting) {
          fprintf(stderr, "No 'stride' in test case %d from %s.\n", i, path);
          continue;
        }
        int stride_count = config_setting_length(stride_setting);
        for (int j = 0; j < stride_count; ++j) {
          int input = config_setting_get_int_elem(stride_setting, j);
          stride.push_back(input);
        }
        config_setting_t* padding_setting =
            config_setting_get_member(test, "padding");
        if (!padding_setting) {
          fprintf(stderr, "No 'padding' in test case %d from %s.\n", i, path);
          continue;
        }
        int padding_count = config_setting_length(padding_setting);
        for (int j = 0; j < padding_count; ++j) {
          int input = config_setting_get_int_elem(padding_setting, j);
          padding.push_back(input);
        }
        config_setting_t* dilation_setting =
            config_setting_get_member(test, "dilation");
        if (!dilation_setting) {
          fprintf(stderr, "No 'dilation' in test case %d from %s.\n", i, path);
          continue;
        }
        int dilation_count = config_setting_length(dilation_setting);
        for (int j = 0; j < dilation_count; ++j) {
          int input = config_setting_get_int_elem(dilation_setting, j);
          dilation.push_back(input);
        }

        Conv_Input tmp(ndim1, dims1, dtype1, device1, nullptr, len1, ndim2,
                       dims2, dtype2, device2, nullptr, len2, stride, padding,
                       dilation, groups);

        inputs.push_back(std::move(tmp));
        inputs_name.emplace_back(input_name);
      }
    } else {
      fprintf(stderr, "Can not find path %s in config.\n", path);
      return (EXIT_FAILURE);
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
      random_assign(input_data1, input_len1, input.dtype1());
      random_assign(input_data2, input_len2, input.dtype2());
      input.set_data1(input_data1, input_len1);
      input.set_data2(input_data2, input_len2);
    }

    config_destroy(&cfg);
    return (EXIT_SUCCESS);
  }

  using InputType = Conv_Input;
  using UserInterface = InterfaceType;
  // inputs
  std::vector<Conv_Input> conv2d_inputs;
  std::vector<std::string> conv2d_inputs_name;
  std::map<std::string, int> test_case = {{"conv2d", 0}};
};
TYPED_TEST_CASE_P(Conv2dTest);

TYPED_TEST_P(Conv2dTest, TwoTests) {
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;

  time_map m;
  auto test = [&m](std::vector<Conv_Input>&& inputs,
                 std::vector<std::string>&& inputs_name,
                 const std::string& test_case_name, int test_case_index) {
    for (int i = 0; i < inputs.size(); i++) {
      // clang-format off
      struct timeval aitisa_start{}, aitisa_end{}, user_start{}, user_end{};
      // clang-format on
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
      gettimeofday(&aitisa_start, nullptr);
      aitisa_conv2d(aitisa_tensor1, aitisa_tensor2, inputs[i].stride(), 2,
                    inputs[i].padding(), 2, inputs[i].dilation(), 2,
                    inputs[i].groups(), &aitisa_result);
      gettimeofday(&aitisa_end, nullptr);
      aitisa_time = (aitisa_end.tv_sec - aitisa_start.tv_sec) * 1000.0 +
                    (aitisa_end.tv_usec - aitisa_start.tv_usec) / 1000.0;
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
      gettimeofday(&user_start, nullptr);
      UserFuncs::user_conv2d(user_tensor1, user_tensor2, inputs[i].stride(), 2,
                             inputs[i].padding(), 2, inputs[i].dilation(), 2,
                             inputs[i].groups(), &user_result);
      gettimeofday(&user_end, nullptr);
      user_time = (user_end.tv_sec - user_start.tv_sec) * 1000.0 +
                  (user_end.tv_usec - user_start.tv_usec) / 1000.0;
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
      // print result of test
      std::cout << /*GREEN <<*/ "[ " << test_case_name << " sample" << i
                << " / " << inputs_name[i] << " ] " << /*RESET <<*/ std::endl;
      std::cout << /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time
                << " ms" << std::endl;
      std::cout << /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time
                << " ms" << std::endl;
      m.insert(std::make_pair(test_case_name+" sample "+std::to_string(i),time_map_value(aitisa_time, user_time)));
    }
  };
  if (this->conv2d_inputs.size()) {
    test(std::move(this->conv2d_inputs), std::move(this->conv2d_inputs_name),
         "conv2d", this->test_case["conv2d"]);
    draw_fig_fun(m,"conv2d");
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(Conv2dTest, TwoTests);

#define REGISTER_CONV2D(CONV_FUNC, CONV)                                       \
  class Conv2d : public Basic {                                                \
   public:                                                                     \
    static void user_conv2d(UserTensor input, UserTensor filter,               \
                            const int* stride, const int stride_len,           \
                            const int* padding, const int padding_len,         \
                            const int* dilation, const int dilation_len,       \
                            const int groups, UserTensor* output) {            \
      typedef std::function<void(const UserTensor, const UserTensor,           \
                                 const int*, const int, const int*, const int, \
                                 const int*, const int, const int,             \
                                 UserTensor*)>                                 \
          conv_func;                                                           \
      auto func_args_num = aitisa_api::function_traits<CONV_FUNC>::nargs;      \
      auto args_num = aitisa_api::function_traits<conv_func>::nargs;           \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<                                                       \
              std::remove_cv<                                                  \
                  aitisa_api::function_traits<conv_func>::result_type>::type,  \
              aitisa_api::function_traits<CONV_FUNC>::result_type>::value) {   \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<aitisa_api::function_traits<conv_func>::nargs,   \
                              conv_func, CONV_FUNC>();                         \
      CONV(input, filter, stride, stride_len, padding, padding_len, dilation,  \
           dilation_len, groups, output);                                      \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, Conv2dTest, Conv2d);               \
  }

}  // namespace aitisa_api