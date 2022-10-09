#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
#include "hice/basic/factories.h"

extern "C" {
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

        std::vector<int64_t> dims, param_dims;
        int test_index, ndim, param_ndim, dtype, device, len, axis;
        std::string input_name;
        float value, mean, var, epsilon;

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
        if (!setting.lookupValue("param_ndim", param_ndim)) {
          std::cerr << "Setting \"param_ndim\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& param_dims_setting =
              setting.lookup("param_dims");
          if (param_dims_setting.getLength() != param_ndim) {
            std::cerr << " \"param_dims\" length is not correct in test index "
                      << test_index << " from " << path << " !\n"
                      << std::endl;
            continue;
          }
          for (int n = 0; n < param_dims_setting.getLength(); ++n) {
            param_dims.push_back((int64_t) int(param_dims_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"param_dims\" do not exist in test index "
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
        if (!setting.lookupValue("value", value)) {
          std::cerr << "Setting \"value\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("mean", mean)) {
          std::cerr << "Setting \"mean\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("var", var)) {
          std::cerr << "Setting \"var\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("epsilon", epsilon)) {
          std::cerr << "Setting \"epsilon\" do not exist in test index "
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
        Batchnorm_Input tmp(ndim, dims, dtype, device, nullptr, len, axis,
                            epsilon, param_ndim, param_dims, value, mean, var);
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
#ifdef AITISA_API_PYTORCH
  using TorchDataType = typename libtorch_api::DataType;
  using TorchDevice = typename libtorch_api::Device;
  using TorchTensor = typename libtorch_api::Tensor;
#endif
  time_map m;
  auto test = [&m](std::vector<Batchnorm_Input>&& inputs,
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
        float *aitisa_result_data = nullptr, *user_result_data = nullptr;
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

        auto aitisa_start = std::chrono::steady_clock::now();

        aitisa_batch_norm(aitisa_tensor, inputs[i].axis(), scale, bias, mean,
                          variance, inputs[i].epsilon(), &aitisa_result);

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
        auto user_start = std::chrono::steady_clock::now();
        UserFuncs::user_batchnorm(
            user_tensor, inputs[i].axis(), bn_scale, bn_bias, running_mean,
            running_var, inputs[i].epsilon(), user_result, bn_mean, bn_var);

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
        TorchTensor torch_bn_mean = torch::full(param_dims,inputs[i].mean());
        TorchTensor torch_bn_var= torch::full(param_dims,inputs[i].var());
        TorchTensor torch_bias= torch::zeros(param_dims);

        torch_result = torch::nn::functional::batch_norm(
            torch_tensor, torch_bn_mean, torch_bn_var,
            torch::nn::functional::BatchNormFuncOptions()
                .bias(torch_bias)
                .momentum(0.1)
                .eps(1e-05)
                .training(false));

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
        ASSERT_EQ(/*CUDA*/ 0,
                  UserFuncs::user_device_to_int(user_result_device));
        ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype),
                  UserFuncs::user_dtype_to_int(user_result_dtype));
        for (int64_t j = 0; j < aitisa_result_ndim; j++) {
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
  if (this->batch_norm_inputs.size()) {
    test(std::move(this->batch_norm_inputs), std::move(this->batch_norm_name),
         "batch_norm", this->test_case["batch_norm"]);
#ifdef AITISA_API_GENERATE_FIGURE
    draw_fig_fun(m, "batch_norm");
#endif
  } else
    FAIL() << "No input test case.";
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
