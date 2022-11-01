#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"

extern "C" {
#include "src/nn/conv.h"
}

namespace aitisa_api {

namespace {

class Conv_Input : public Binary_Input {
 public:
  Conv_Input() = default;
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
        int test_index, ndim1, ndim2, dtype1, device1, len1, dtype2, device2,
            len2, groups;
        std::string input_name;
        std::vector<int> stride, padding, dilation;

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
        if (!setting.lookupValue("groups", groups)) {
          std::cerr << "Setting \"groups\" do not exist in test index "
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

        Conv_Input tmp(ndim1, dims1, dtype1, device1, nullptr, len1, ndim2,
                       dims2, dtype2, device2, nullptr, len2, stride, padding,
                       dilation, groups);
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
        void* input_data1 = (void*)new char[input_len1];
        void* input_data2 = (void*)new char[input_len2];
        random_assign(input_data1, input_len1, input.dtype1());
        random_assign(input_data2, input_len2, input.dtype2());
        input.set_data1(input_data1, input_len1);
        input.set_data2(input_data2, input_len2);
      }
    } catch (const libconfig::SettingNotFoundException& nfex) {
      std::cerr << nfex.getPath() << " do not exist! " << std::endl;
      return (EXIT_FAILURE);
    }
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
#ifdef AITISA_API_PYTORCH
  using TorchDataType = typename libtorch_api::DataType;
  using TorchDevice = typename libtorch_api::Device;
  using TorchTensor = typename libtorch_api::Tensor;
#endif

  time_map m;
  auto test = [&m](std::vector<Conv_Input>&& inputs,
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
        AITISA_Tensor aitisa_tensor1, aitisa_tensor2, aitisa_result;
        AITISA_DataType aitisa_result_dtype;
        AITISA_Device aitisa_result_device;
        UserTensor user_tensor1, user_tensor2, user_result;
        UserDataType user_result_dtype;
        UserDevice user_result_device;
#ifdef AITISA_API_PYTORCH
        int64_t torch_result_ndim;
        int64_t* torch_result_dims = nullptr;
        float* torch_result_data = nullptr;
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
        aitisa_conv2d(aitisa_tensor1, aitisa_tensor2, inputs[i].stride(), 2,
                      inputs[i].padding(), 2, inputs[i].dilation(), 2,
                      inputs[i].groups(), &aitisa_result);
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
        UserFuncs::user_conv2d(user_tensor1, user_tensor2, inputs[i].stride(),
                               2, inputs[i].padding(), 2, inputs[i].dilation(),
                               2, inputs[i].groups(), &user_result);
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
        std::vector<int64_t> stride_list, padding_list, dilation_list;

        for (int index = 0; index < 2; index++) {
          stride_list.push_back(inputs[i].stride()[index]);
          padding_list.push_back(inputs[i].padding()[index]);
          dilation_list.push_back(inputs[i].dilation()[index]);
        }
        torch_result = torch::nn::functional::conv2d(
            torch_tensor1, torch_tensor2,
            torch::nn::functional::Conv2dFuncOptions()
                .stride(stride_list)
                .padding(padding_list)
                .dilation(dilation_list)
                .groups(inputs[i].groups()));

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
          ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
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
  if (this->conv2d_inputs.size()) {
    test(std::move(this->conv2d_inputs), std::move(this->conv2d_inputs_name),
         "conv2d", this->test_case["conv2d"]);
#ifdef AITISA_API_GENERATE_FIGURE
    draw_fig_fun(m, "conv2d");
#endif
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