#pragma once

#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
#include "hice/basic/factories.h"

extern "C" {
#include <math.h>
#include <sys/time.h>
#include "src/nn/batch_norm.h"
}

namespace aitisa_api {

namespace {

class Batchnorm_Input : public Unary_Input {
 public:
  Batchnorm_Input() = default;
  Batchnorm_Input(int64_t ndim, std::vector<int64_t> dims, int dtype,
                  int device, void* data, unsigned int len, int axis,
                  double epsilon, int64_t param_ndim,
                  std::vector<int64_t> param_dims, float value, float mean,
                  float var)
      : Unary_Input(ndim, std::move(dims), dtype, device, data, len),
        axis_(axis),
        epsilon_(epsilon),
        param_ndim_(param_ndim),
        param_dims_(nullptr),
        value_(value),
        mean_(mean),
        var_(var) {
    int64_t spatial_len = this->param_ndim_;
    this->param_dims_ = new int64_t[spatial_len];
    for (int64_t i = 0; i < spatial_len; i++) {
      this->param_dims_[i] = param_dims[i];
    }
  }
  Batchnorm_Input(Batchnorm_Input&& input) noexcept
      : Unary_Input(input.ndim(), input.dims(), input.dtype(), input.device(),
                    input.data(), input.len()),
        axis_(input.axis()),
        epsilon_(input.epsilon()),
        param_ndim_(input.param_ndim()),
        param_dims_(input.param_dims()),
        value_(input.value()),
        mean_(input.mean()),
        var_(input.var()) {
    input.to_nullptr();
    input.param_dims_ = nullptr;
  };

  ~Batchnorm_Input() override { delete[] param_dims_; }
  Batchnorm_Input& operator=(Batchnorm_Input& right) {
    int64_t spatial_len = right.param_ndim();
    auto& left = (Unary_Input&)(*this);
    left = (Unary_Input&)right;
    this->axis_ = right.axis();
    this->epsilon_ = right.epsilon();
    this->param_ndim_ = right.param_ndim();
    this->value_ = right.value();
    this->mean_ = right.mean();
    this->var_ = right.var();
    this->param_dims_ = new int64_t[spatial_len];
    memcpy(this->param_dims_, right.param_dims(),
           spatial_len * sizeof(int64_t));
  }

  int axis() const { return axis_; }
  double epsilon() const { return epsilon_; }
  int64_t param_ndim() const { return param_ndim_; }
  int64_t* param_dims() { return param_dims_; }
  float value() const { return value_; }
  float mean() const { return mean_; }
  float var() const { return var_; }

 private:
  int axis_ = 0;
  double epsilon_ = 0.0;
  int64_t param_ndim_ = 0;
  int64_t* param_dims_ = nullptr;
  float value_{};
  float mean_{};
  float var_{};
};

}  // namespace

template <typename InterfaceType>
class BatchnormTest : public ::testing::Test {
 public:
  BatchnormTest() {
    fetch_test_data("batch_norm", batch_norm_inputs, batch_norm_name);
  }
  ~BatchnormTest() override = default;
  static void aitisa_kernel(const Tensor input, const int axis,
                            const Tensor scale, const Tensor bias,
                            const Tensor mean, const Tensor variance,
                            const double epsilon, Tensor* output) {
    aitisa_batch_norm(input, axis, scale, bias, mean, variance, epsilon,
                      output);
  }
  int fetch_test_data(const char* path, std::vector<Batchnorm_Input>& inputs,
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
        config_setting_t* param_dims_setting =
            config_setting_lookup(test, "param_dims");
        if (!param_dims_setting) {
          fprintf(stderr, "No 'param_dims' in test case %d from %s.\n", i, path);
          continue;
        }
        int64_t ndim, param_ndim;
        std::vector<int64_t> dims, param_dims;
        int dtype, device, len, axis;
        const char* input_name;
        double value, mean, var;
        double epsilon;

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
        if (!config_setting_lookup_int64(
                test, "param_ndim",
                reinterpret_cast<long long int*>(&param_ndim))) {
          fprintf(stderr, "No 'param_ndim' in test case %d from %s.\n", i,
                  path);
          continue;
        }
        if (config_setting_length(param_dims_setting) != param_ndim) {
          fprintf(stderr,
                  "'param_dims' length is not correct in test case %d from %s.\n", i,
                  path);
          continue;
        }
        for (int j = 0; j < ndim; ++j) {
          int64_t input = config_setting_get_int_elem(dims_setting, j);
          dims.push_back(input);
        }
        for (int k = 0; k < param_ndim; ++k) {
          int64_t input = config_setting_get_int_elem(param_dims_setting, k);
          param_dims.push_back(input);
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
        if (!config_setting_lookup_float(test, "epsilon", &epsilon)) {
          fprintf(stderr, "No 'epsilon' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_float(test, "value",
                                         reinterpret_cast<double*>(&value))) {
          fprintf(stderr, "No 'value' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_float(test, "mean",
                                         reinterpret_cast<double*>(&mean))) {
          fprintf(stderr, "No 'mean' in test case %d from %s.\n", i, path);
          continue;
        }
        if (!config_setting_lookup_float(test, "var",
                                         reinterpret_cast<double*>(&var))) {
          fprintf(stderr, "No 'var' in test case %d from %s.\n", i, path);
          continue;
        }

        Batchnorm_Input tmp(
            /*ndim*/ ndim, /*dims*/ dims, /*dtype=double*/ dtype,
            /*device=cpu*/ device, /*data*/ nullptr, /*len*/ len, axis, epsilon,
            param_ndim, param_dims, value, mean, var);

        inputs.push_back(std::move(tmp));
        inputs_name.emplace_back(input_name);
      }
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

  using InputType = Batchnorm_Input;
  using UserInterface = InterfaceType;
  // inputs
  std::vector<Batchnorm_Input> batch_norm_inputs;
  std::vector<std::string> batch_norm_name;
  std::map<std::string, int> test_case = {{"batch_norm", 0}};
};
TYPED_TEST_CASE_P(BatchnormTest);

TYPED_TEST_P(BatchnormTest, TwoTests) {
  using UserDataType = typename TestFixture::UserInterface::UserDataType;
  using UserDevice = typename TestFixture::UserInterface::UserDevice;
  using UserTensor = typename TestFixture::UserInterface::UserTensor;
  using UserFuncs = typename TestFixture::UserInterface;

  auto test = [](std::vector<Batchnorm_Input>&& inputs,
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
      full_float(aitisa_tensor, inputs[i].value());

      AITISA_Tensor mean, variance, scale, bias;

      aitisa_create(aitisa_dtype, aitisa_device, inputs[i].param_dims(),
                    inputs[i].param_ndim(), nullptr, 0, &mean);
      full_float(mean, inputs[i].mean());

      aitisa_create(aitisa_dtype, aitisa_device, inputs[i].param_dims(),
                    inputs[i].param_ndim(), nullptr, 0, &variance);
      full_float(variance, inputs[i].var());

      aitisa_create(aitisa_dtype, aitisa_device, inputs[i].param_dims(),
                    inputs[i].param_ndim(), nullptr, 0, &scale);
      full_float(scale, 1);

      aitisa_create(aitisa_dtype, aitisa_device, inputs[i].param_dims(),
                    inputs[i].param_ndim(), nullptr, 0, &bias);
      full_float(bias, 0);

      gettimeofday(&aitisa_start, nullptr);

      aitisa_batch_norm(aitisa_tensor, inputs[i].axis(), scale, bias, mean,
                        variance, inputs[i].epsilon(), &aitisa_result);

      gettimeofday(&aitisa_end, nullptr);
      aitisa_time = (aitisa_end.tv_sec - aitisa_start.tv_sec) * 1000.0 +
                    (aitisa_end.tv_usec - aitisa_start.tv_usec) / 1000.0;
      aitisa_resolve(aitisa_result, &aitisa_result_dtype, &aitisa_result_device,
                     &aitisa_result_dims, &aitisa_result_ndim,
                     (void**)&aitisa_result_data, &aitisa_result_len);

      // user
      UserDataType user_dtype =
          UserFuncs::user_int_to_dtype(inputs[i].dtype());
      UserDevice user_device =
          UserFuncs::user_int_to_device(inputs[i].device());
      UserFuncs::user_create(user_dtype, user_device, inputs[i].dims(),
                             inputs[i].ndim(), inputs[i].data(),
                             inputs[i].len(), &user_tensor);

      gettimeofday(&user_start, nullptr);
      std::vector<int64_t> param_dims = {};
      for (int k = 0; k < inputs[i].param_ndim(); k++) {
        param_dims.push_back(inputs[i].param_dims()[k]);
      }
      UserTensor bn_scale =
          hice::empty(param_dims, device(hice::kCPU).dtype(hice::kFloat));
      UserTensor bn_bias =
          hice::empty(param_dims, device(hice::kCPU).dtype(hice::kFloat));
      UserTensor running_mean =
          hice::empty(param_dims, device(hice::kCPU).dtype(hice::kFloat));
      UserTensor running_var =
          hice::empty(param_dims, device(hice::kCPU).dtype(hice::kFloat));

      UserTensor bn_mean = hice::full(param_dims, inputs[i].mean(),
                                      device(hice::kCPU).dtype(hice::kFloat));
      UserTensor bn_var = hice::full(param_dims, inputs[i].var(),
                                     device(hice::kCPU).dtype(hice::kFloat));

      UserFuncs::user_create(user_dtype, user_device, inputs[i].dims(),
                             inputs[i].ndim(), NULL, inputs[i].len(),
                             &user_result);

      UserFuncs::user_batchnorm(
          user_tensor, inputs[i].axis(), bn_scale, bn_bias, running_mean,
          running_var, inputs[i].epsilon(), user_result, bn_mean, bn_var);

      gettimeofday(&user_end, nullptr);
      user_time = (user_end.tv_sec - user_start.tv_sec) * 1000.0 +
                  (user_end.tv_usec - user_start.tv_usec) / 1000.0;
      UserFuncs::user_resolve(
          user_result, &user_result_dtype, &user_result_device, &user_result_dims,
          &user_result_ndim, (void**)&user_result_data, &user_result_len);
      // compare
      int64_t tensor_size = 1;
      ASSERT_EQ(aitisa_result_ndim, user_result_ndim);
      ASSERT_EQ(/*CUDA*/ 0, UserFuncs::user_device_to_int(user_result_device));
      ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype),
                UserFuncs::user_dtype_to_int(user_result_dtype));
      for (int64_t j = 0; j < aitisa_result_ndim; j++) {
        ASSERT_EQ(aitisa_result_dims[j], user_result_dims[j]);
      }
      ASSERT_EQ(aitisa_result_len, user_result_len);
      auto* aitisa_data = (float*)aitisa_result_data;
      auto* user_data = (float*)user_result_data;
      for (int64_t j = 0; j < tensor_size; j++) {
        ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
      }
      // print result of test
      std::cout << /*GREEN <<*/ "[ " << test_case_name << " sample" << i << " / "
                << inputs_name[i] << " ] " << /*RESET <<*/ std::endl;
      std::cout << /*GREEN <<*/ "\t[ AITISA ] " << /*RESET <<*/ aitisa_time
                << " ms" << std::endl;
      std::cout << /*GREEN <<*/ "\t[  USER  ] " << /*RESET <<*/ user_time << " ms"
                << std::endl;
    }
  };
  test(std::move(this->batch_norm_inputs), std::move(this->batch_norm_name), "batch_norm",
       this->test_case["batch_norm"]);
}
REGISTER_TYPED_TEST_CASE_P(BatchnormTest, TwoTests);

#define REGISTER_BATCHNORM(BATCHNORM_FUNC, BATCHNORM)                          \
  class Batchnorm : public Basic {                                             \
   public:                                                                     \
    static void user_batchnorm(UserTensor input, const int axis,               \
                               UserTensor scale, UserTensor bias,              \
                               UserTensor running_mean,                        \
                               UserTensor running_variance,                    \
                               const double epsilon, UserTensor output,        \
                               UserTensor mean, UserTensor var) {              \
      typedef std::function<void(UserTensor&, int, UserTensor&, UserTensor&,   \
                                 UserTensor&, UserTensor&, double,             \
                                 UserTensor&, UserTensor&, UserTensor&)>       \
          batchnorm_func;                                                      \
      auto func_args_num = aitisa_api::function_traits<BATCHNORM_FUNC>::nargs; \
      auto args_num = aitisa_api::function_traits<batchnorm_func>::nargs;      \
      if (func_args_num != args_num) {                                         \
        throw std::invalid_argument("Incorrect parameter numbers: expected " + \
                                    std::to_string(args_num) +                 \
                                    " arguments but got " +                    \
                                    std::to_string(func_args_num));            \
      }                                                                        \
      if (!std::is_same<std::remove_cv<aitisa_api::function_traits<            \
                            batchnorm_func>::result_type>::type,               \
                        aitisa_api::function_traits<                           \
                            BATCHNORM_FUNC>::result_type>::value) {            \
        throw std::invalid_argument(                                           \
            "Incorrect return type: type mismatch at return");                 \
      }                                                                        \
      aitisa_api::TypeCompare<                                                 \
          aitisa_api::function_traits<batchnorm_func>::nargs, batchnorm_func,  \
          BATCHNORM_FUNC>();                                                   \
      BATCHNORM(input, axis, scale, bias, running_mean, running_variance,      \
                epsilon, output, mean, var);                                   \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, BatchnormTest, Batchnorm);         \
  }

}  // namespace aitisa_api
