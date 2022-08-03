#pragma once

#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"

extern "C" {
#include <math.h>
#include <sys/time.h>
#include "src/nn/pooling.h"
}

namespace aitisa_api {

namespace {

class Pooling_Input : public Unary_Input {
 public:
  Pooling_Input() = default;
  ;
  Pooling_Input(int64_t ndim, int64_t* dims, int dtype, int device, void* data,
                unsigned int len, int* stride, int* padding, int* dilation,
                int* ksize, char* mode)
      : Unary_Input(ndim, dims, dtype, device, data, len),
        stride_(stride),
        padding_(padding),
        dilation_(dilation),
        ksize_(ksize),
        mode_(mode) {}
  Pooling_Input(int64_t ndim, std::vector<int64_t> dims, int dtype, int device,
                void* data, unsigned int len, std::vector<int> stride,
                std::vector<int> padding, std::vector<int> dilation,
                std::vector<int> ksize, std::string mode)
      : Unary_Input(ndim, dims, dtype, device, data, len),
        stride_(nullptr),
        padding_(nullptr),
        dilation_(nullptr),
        ksize_(nullptr),
        mode_(nullptr) {
    int spatial_len = ndim - 2;
    this->stride_ = new int[spatial_len];
    this->padding_ = new int[spatial_len];
    this->dilation_ = new int[spatial_len];
    this->ksize_ = new int[spatial_len];
    this->mode_ = new char[4];
    for (int i = 0; i < 3; i++) {
      this->mode_[i] = mode[i];
    }
    mode_[3] = '\0';
    for (int i = 0; i < spatial_len; i++) {
      this->stride_[i] = stride[i];
      this->padding_[i] = padding[i];
      this->dilation_[i] = dilation[i];
      this->ksize_[i] = ksize[i];
    }
  }
  Pooling_Input(Pooling_Input&& input) noexcept
      : Unary_Input(input.ndim(), input.dims(), input.dtype(), input.device(),
                    input.data(), input.len()),
        stride_(input.stride()),
        padding_(input.padding()),
        dilation_(input.dilation()),
        ksize_(input.ksize()),
        mode_(input.mode()) {
    input.to_nullptr();
    input.stride_ = nullptr;
    input.padding_ = nullptr;
    input.dilation_ = nullptr;
    input.ksize_ = nullptr;
    input.mode_ = nullptr;
  }
  ~Pooling_Input() override {
    delete[] stride_;
    delete[] padding_;
    delete[] dilation_;
    delete[] ksize_;
    delete[] mode_;
  }
  Pooling_Input& operator=(Pooling_Input& right) {
    int spatial_len = right.ndim() - 2;
    Unary_Input& left = (Unary_Input&)(*this);
    left = (Unary_Input&)right;
    this->stride_ = new int[spatial_len];
    this->padding_ = new int[spatial_len];
    this->dilation_ = new int[spatial_len];
    this->ksize_ = new int[spatial_len];
    this->mode_ = new char[3];
    memcpy(this->stride_, right.stride(), spatial_len * sizeof(int));
    memcpy(this->padding_, right.padding(), spatial_len * sizeof(int));
    memcpy(this->dilation_, right.dilation(), spatial_len * sizeof(int));
    memcpy(this->ksize_, right.ksize(), spatial_len * sizeof(int));
    memcpy(this->mode_, right.mode(), 3 * sizeof(char));
  }
  int* stride() { return stride_; }
  int* padding() { return padding_; }
  int* dilation() { return dilation_; }
  int* ksize() { return ksize_; }
  char* mode() { return mode_; }

 private:
  int* stride_ = nullptr;
  int* padding_ = nullptr;
  int* dilation_ = nullptr;
  int* ksize_ = nullptr;
  char* mode_ = nullptr;
};

}  // namespace

template <typename InterfaceType>
class PoolingTest : public ::testing::Test {
 public:
  PoolingTest() { fetch_test_data("pooling", pooling_inputs, pooling_name); }
  ~PoolingTest() override = default;
  using InputType = Pooling_Input;
  using UserInterface = InterfaceType;
  static void aitisa_kernel(const AITISA_Tensor input, const char* mode,
                            const int* ksize, const int* stride,
                            const int* padding, const int* dilation,
                            AITISA_Tensor* output) {
    aitisa_pooling(input, mode, ksize, stride, padding, dilation, output);
  }

  int fetch_test_data(const char* path, std::vector<Pooling_Input>& inputs,
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
        std::vector<int> stride, padding, dilation, ksize;
        int dtype, device, len;
        const char* input_name;
        const char* mod;

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
        config_setting_t* ksize_setting =
            config_setting_get_member(test, "ksize");
        if (!ksize_setting) {
          fprintf(stderr, "No 'ksize' in test case %d from %s.\n", i, path);
          continue;
        }
        int ksize_count = config_setting_length(ksize_setting);
        for (int j = 0; j < ksize_count; ++j) {
          int input = config_setting_get_int_elem(ksize_setting, j);
          ksize.push_back(input);
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
        if (!config_setting_lookup_string(test, "mod", &mod)) {
          fprintf(stderr, "No 'mod' in test case %d from %s.\n", i, path);
          continue;
        }

        Pooling_Input tmp(
            /*ndim*/ ndim, /*dims*/ dims, /*dtype=double*/ dtype,
            /*device=cpu*/ device, /*data*/ nullptr, /*len*/ len, stride,
            padding, dilation, ksize, mod);

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
  std::vector<Pooling_Input> pooling_inputs;
  std::vector<std::string> pooling_name;
  std::map<std::string, int> test_case = {{"pooling", 0}};
};
TYPED_TEST_CASE_P(PoolingTest);

TYPED_TEST_P(PoolingTest, TwoTests) {
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;

  auto test = [](std::vector<Pooling_Input>&& inputs,
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
      gettimeofday(&aitisa_start, nullptr);

      aitisa_pooling(aitisa_tensor, inputs[i].mode(), inputs[i].ksize(),
                     inputs[i].stride(), inputs[i].padding(),
                     inputs[i].dilation(), &aitisa_result);

      gettimeofday(&aitisa_end, nullptr);
      aitisa_time = (aitisa_end.tv_sec - aitisa_start.tv_sec) * 1000.0 +
                    (aitisa_end.tv_usec - aitisa_start.tv_usec) / 1000.0;
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
      gettimeofday(&user_start, nullptr);
      UserFuncs::user_pooling(user_tensor, inputs[i].stride(), 2,
                              inputs[i].padding(), 2, inputs[i].dilation(), 2,
                              inputs[i].ksize(), 2, inputs[i].mode(), 3,
                              &user_result);
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
    }
  };
  if (this->pooling_inputs.size()) {
    test(std::move(this->pooling_inputs), std::move(this->pooling_name),
         "pooling", this->test_case["pooling"]);
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(PoolingTest, TwoTests);

#define REGISTER_POOLING(POOLING_FUNC, POOLING)                                \
  class Pooling : public Basic {                                               \
   public:                                                                     \
    static void user_pooling(UserTensor input, const int* stride,              \
                             const int stride_len, const int* padding,         \
                             const int padding_len, const int* dilation,       \
                             const int dilation_len, const int* ksize,         \
                             const int ksize_len, const char* mode,            \
                             const int mode_len, UserTensor* output) {         \
      typedef std::function<void(const UserTensor, const int*, const int,      \
                                 const int*, const int, const int*, const int, \
                                 const int*, const int, const char*,           \
                                 const int, UserTensor*)>                      \
          pooling_func;                                                        \
      auto func_args_num = aitisa_api::function_traits<POOLING_FUNC>::nargs;   \
      auto args_num = aitisa_api::function_traits<pooling_func>::nargs;        \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<std::remove_cv<aitisa_api::function_traits<            \
                            pooling_func>::result_type>::type,                 \
                        aitisa_api::function_traits<                           \
                            POOLING_FUNC>::result_type>::value) {              \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<                                                 \
          aitisa_api::function_traits<pooling_func>::nargs, pooling_func,      \
          POOLING_FUNC>();                                                     \
      POOLING(input, stride, stride_len, padding, padding_len, dilation,       \
              dilation_len, ksize, ksize_len, mode, mode_len, output);         \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, PoolingTest, Pooling);             \
  }

}  // namespace aitisa_api