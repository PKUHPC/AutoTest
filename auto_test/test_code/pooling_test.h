#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include "auto_test/basic.h"
#include "auto_test/sample.h"

extern "C" {
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
        std::vector<int> stride, padding, dilation, ksize;
        int test_index, ndim, dtype, device, len;
        std::string input_name;
        std::string mod;

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
        try {
          const libconfig::Setting& stride_setting = setting.lookup("stride");
          for (int n = 0; n < stride_setting.getLength(); ++n) {
            stride.push_back(int(stride_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"stride\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& padding_setting = setting.lookup("padding");
          for (int n = 0; n < padding_setting.getLength(); ++n) {
            padding.push_back(int(padding_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"padding\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& dilation_setting =
              setting.lookup("dilation");
          for (int n = 0; n < dilation_setting.getLength(); ++n) {
            dilation.push_back(int(dilation_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"stride\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& ksize_setting = setting.lookup("ksize");
          for (int n = 0; n < ksize_setting.getLength(); ++n) {
            ksize.push_back(int(ksize_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"ksize\" do not exist in test index "
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
        if (!setting.lookupValue("mod", mod)) {
          std::cerr << "Setting \"mod\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        Pooling_Input tmp(ndim, dims, dtype, device, nullptr, len, stride,
                          padding, dilation, ksize, mod);
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
#ifdef AITISA_API_PYTORCH
  using TorchDataType = typename libtorch_api::DataType;
  using TorchDevice = typename libtorch_api::Device;
  using TorchTensor = typename libtorch_api::Tensor;
#endif
  time_map m;
  auto test = [&m](std::vector<Pooling_Input>&& inputs,
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
        auto aitisa_start = std::chrono::steady_clock::now();
        aitisa_pooling(aitisa_tensor, inputs[i].mode(), inputs[i].ksize(),
                       inputs[i].stride(), inputs[i].padding(),
                       inputs[i].dilation(), &aitisa_result);

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
        UserFuncs::user_pooling(user_tensor, inputs[i].stride(), 2,
                                inputs[i].padding(), 2, inputs[i].dilation(), 2,
                                inputs[i].ksize(), 2, inputs[i].mode(), 3,
                                &user_result);
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
        std::vector<int64_t> ksize_list, stride_list, padding_list,
            dilation_list;

        for (int index = 0; index < 2; index++) {
          ksize_list.push_back(inputs[i].ksize()[index]);
          stride_list.push_back(inputs[i].stride()[index]);
          padding_list.push_back(inputs[i].padding()[index]);
          dilation_list.push_back(inputs[i].dilation()[index]);
        }
        std::string mode_str(inputs[i].mode());
        if (mode_str == "avg") {
          torch_result = torch::nn::functional::avg_pool2d(
              torch_tensor,
              torch::nn::functional::AvgPool2dFuncOptions(ksize_list)
                  .stride(stride_list)
                  .padding(padding_list));
        } else if (mode_str == "max") {
          torch_result = torch::nn::functional::max_pool2d(
              torch_tensor,
              torch::nn::functional::MaxPool2dFuncOptions(ksize_list)
                  .stride(stride_list)
                  .padding(padding_list));
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
#endif
      m.insert(std::make_pair(test_case_name + " sample " + std::to_string(i),
                              time_map_value(aitisa_time, user_time)));
    }
  };
  if (this->pooling_inputs.size()) {
    test(std::move(this->pooling_inputs), std::move(this->pooling_name),
         "pooling", this->test_case["pooling"]);
#ifdef AITISA_API_GENERATE_FIGURE
    draw_fig_fun(m, "pooling");
#endif
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