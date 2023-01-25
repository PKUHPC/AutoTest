#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
extern "C" {
#include "src/basic/factories.h"
#include "src/new_ops6/l1_loss.h"
}

namespace aitisa_api {

namespace {

class L1LossInput : public Binary_Input {
 public:
  L1LossInput() = default;
  L1LossInput(int64_t ndim1, std::vector<int64_t> dims1, int dtype1,
              int device1, void* data1, unsigned int len1, int64_t ndim2,
              std::vector<int64_t> dims2, int dtype2, int device2, void* data2,
              unsigned int len2, std::vector<int> weight, int64_t reduction)
      : Binary_Input(ndim1, std::move(dims1), dtype1, device1, data1, len1,
                     ndim2, std::move(dims2), dtype2, device2, data2, len2),
        weight_(nullptr),
        reduction_(reduction) {
    int spatial_len = ndim1;
    this->weight_ = new int[spatial_len];
    for (int i = 0; i < spatial_len; i++) {
      this->weight_[i] = weight[i];
    }
  }
  L1LossInput(L1LossInput&& input) noexcept
      : Binary_Input(input.ndim1(), input.dims1(), input.dtype1(),
                     input.device1(), input.data1(), input.len1(),
                     input.ndim2(), input.dims2(), input.dtype2(),
                     input.device2(), input.data2(), input.len2()),
        weight_(input.weight()),
        reduction_(input.reduction()) {
    input.to_nullptr();
    input.weight_ = nullptr;
  }
  ~L1LossInput() override { delete[] weight_; }
  L1LossInput& operator=(L1LossInput&& right) noexcept {
    int spatial_len = right.ndim1();
    auto& left = (Binary_Input&)(*this);
    left = (Binary_Input&)right;
    this->weight_ = new int[spatial_len];
    this->reduction_ = right.reduction();
    memcpy(this->weight_, right.weight(), spatial_len * sizeof(int));
  }
  int* weight() { return weight_; }
  int64_t reduction() const { return reduction_; }

 private:
  int* weight_ = nullptr;
  int64_t reduction_ = 1;
};

}  // namespace

template <typename InterfaceType>
class L1LossTest : public ::testing::Test {
 public:
  L1LossTest() {
    fetch_test_data("l1_loss", l1_loss_inputs, l1_loss_inputs_name);
  }
  ~L1LossTest() override = default;

  int fetch_test_data(const char* path, std::vector<L1LossInput>& inputs,
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
            len2, reduction;
        std::string input_name;
        std::vector<int> weight;

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
        if (!setting.lookupValue("reduction", reduction)) {
          std::cerr << "Setting \"reduction\" do not exist in test index "
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
          const libconfig::Setting& weight_setting = setting.lookup("weight");
          for (int n = 0; n < weight_setting.getLength(); ++n) {
            weight.push_back(int(weight_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"weight\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }

        L1LossInput tmp(ndim1, dims1, dtype1, device1, nullptr, len1, ndim2,
                        dims2, dtype2, device2, nullptr, len2, weight,
                        reduction);
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
  using InputType = L1LossInput;
  using UserInterface = InterfaceType;
  // inputs
  std::vector<L1LossInput> l1_loss_inputs;
  std::vector<std::string> l1_loss_inputs_name;
  std::map<std::string, int> test_case = {{"l1_loss", 0}};
};
TYPED_TEST_CASE_P(L1LossTest);

TYPED_TEST_P(L1LossTest, TwoTests) {
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
  auto test = [&m](std::vector<L1LossInput>&& inputs,
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
        AITISA_Tensor aitisa_tensor1, aitisa_tensor2, aitisa_result;
        AITISA_DataType aitisa_result_dtype;
        AITISA_Device aitisa_result_device;
        UserTensor user_tensor1, user_tensor2, user_result;
        UserDataType user_result_dtype;
        UserDevice user_result_device;
#ifdef AITISA_API_PYTORCH
        int64_t torch_result_ndim;
        int64_t* torch_result_dims = nullptr;
        void* torch_result_data = nullptr;
        unsigned int torch_result_len;
        TorchTensor torch_tensor1, torch_tensor2, torch_result;
        TorchDataType torch_result_dtype;
        TorchDevice torch_result_device(c10::DeviceType::CPU);
#endif
        // aitisa
        AITISA_DataType aitisa_dtype1 = aitisa_int_to_dtype(inputs[i].dtype1());
        AITISA_DataType aitisa_dtype2 = aitisa_int_to_dtype(inputs[i].dtype2());
        AITISA_Device aitisa_device1 = aitisa_int_to_device(0);
        AITISA_Device aitisa_device2 = aitisa_int_to_device(0);
        aitisa_create(aitisa_dtype1, aitisa_device1, inputs[i].dims1(),
                      inputs[i].ndim1(), (void*)(inputs[i].data1()),
                      inputs[i].len1(), &aitisa_tensor1);
        aitisa_create(aitisa_dtype2, aitisa_device2, inputs[i].dims2(),
                      inputs[i].ndim2(), (void*)(inputs[i].data2()),
                      inputs[i].len2(), &aitisa_tensor2);
        Tensor aitisa_weight;
        aitisa_full(aitisa_dtype1, aitisa_device1, inputs[i].dims1(),
                    inputs[i].ndim1(), 1, &aitisa_weight);
        auto aitisa_start = std::chrono::steady_clock::now();

        aitisa_l1_loss(aitisa_tensor1, aitisa_tensor2, aitisa_weight,
                       inputs[i].reduction(), &aitisa_result);

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

        user_result = UserFuncs::user_l1_loss(
            user_tensor1, user_tensor2, inputs[i].weight(),
            inputs[i].reduction(), inputs[i].ndim1());

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
        std::vector<int64_t> weight_list;

        for (int index = 0; index < inputs[i].ndim1(); index++) {
          weight_list.push_back(inputs[i].weight()[index]);
        }
        torch_result =
            torch::l1_loss(torch_tensor1, torch_tensor2,
                           (inputs[i].reduction() == 1 ? at::Reduction::Mean
                                                       : at::Reduction::Sum));

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
        auto* aitisa_data = (double*)aitisa_result_data;
        auto* user_data = (double*)user_result_data;
#ifdef AITISA_API_PYTORCH
        auto* torch_data = (double*)torch_result_data;
        for (int64_t j = 0; j < tensor_size; j++) {
          ASSERT_TRUE(abs(aitisa_data[j] - torch_data[j]) < 1e-3);
        }
#endif
        for (int64_t j = 0; j < tensor_size; j++) {
          ASSERT_TRUE(abs(aitisa_data[j] - user_data[j]) < 1e-3);
        }
        aitisa_tensor1->storage->data = nullptr;
        aitisa_tensor2->storage->data = nullptr;
        aitisa_destroy(&aitisa_tensor1);
        aitisa_destroy(&aitisa_tensor2);
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
  if (this->l1_loss_inputs.size()) {
    test(std::move(this->l1_loss_inputs), std::move(this->l1_loss_inputs_name),
         "l1_loss", this->test_case["l1_loss"]);
#ifdef AITISA_API_GENERATE_FIGURE
    draw_fig_fun(m, "l1_loss");
#endif
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(L1LossTest, TwoTests);

#define REGISTER_L1LOSS(L1LOSS)                                                \
  class L1Loss : public Basic {                                                \
   public:                                                                     \
    static UserTensor user_l1_loss(UserTensor input, UserTensor target,        \
                                   const int* weight, const int64_t reduction, \
                                   const int stride_len) {                     \
                                                                               \
      std::vector<int64_t> weight_vector = {};                                 \
                                                                               \
      weight_vector.reserve(stride_len);                                       \
      for (auto i = 0; i < stride_len; i++) {                                  \
        weight_vector.push_back(weight[i]);                                    \
      }                                                                        \
      hice::Tensor weight_one = full(                                          \
          weight_vector, 1.0, hice::device(hice::kCPU).dtype(hice::kDouble));  \
                                                                               \
      return L1LOSS(                                                           \
          input, target, weight_one,                                           \
          (reduction == 1 ? hice::Reduction::mean : hice::Reduction::sum));    \
    }                                                                          \
  };                                                                           \
  namespace aitisa_api {                                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, L1LossTest, L1Loss);               \
  }

}  // namespace aitisa_api