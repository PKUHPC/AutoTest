#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"

extern "C" {
#include "src/nn/attention.h"
}

namespace aitisa_api {

namespace {

class Attention_Input : public Ternary_Input {
 public:
  Attention_Input() = default;

  Attention_Input(int64_t ndim1, std::vector<int64_t> dims1, int dtype1,
                  int device1, void* data1, unsigned int len1, int64_t ndim2,
                  std::vector<int64_t> dims2, int dtype2, int device2,
                  void* data2, unsigned int len2, int64_t ndim3,
                  std::vector<int64_t> dims3, int dtype3, int device3,
                  void* data3, unsigned int len3, int64_t is_causal)
      : Ternary_Input(ndim1, std::move(dims1), dtype1, device1, data1, len1,
                      ndim2, std::move(dims2), dtype2, device2, data2, len2,
                      ndim3, std::move(dims3), dtype3, device3, data3, len3),
        is_causal_(is_causal) {}

  Attention_Input(Attention_Input&& input) noexcept
      : Ternary_Input(input.ndim1(), input.dims1(), input.dtype1(),
                      input.device1(), input.data1(), input.len1(),
                      input.ndim2(), input.dims2(), input.dtype2(),
                      input.device2(), input.data2(), input.len2(),
                      input.ndim3(), input.dims3(), input.dtype3(),
                      input.device3(), input.data3(), input.len3()),
        is_causal_(input.is_causal()) {
    input.to_nullptr();
  }
  ~Attention_Input() override = default;

  Attention_Input& operator=(Attention_Input& right) {
    auto& left = (Ternary_Input&)(*this);
    left = (Ternary_Input&)right;
    this->is_causal_ = right.is_causal();
  }
  int is_causal() const { return is_causal_; }

 private:
  int is_causal_ = 0;
};
}  // namespace

template <typename InterfaceType>
class AttentionTest : public ::testing::Test {
 public:
  AttentionTest() {
    fetch_test_data("attention", attention_inputs, attention_name);
  }
  ~AttentionTest() override = default;
  int fetch_test_data(const char* path, std::vector<Attention_Input>& inputs,
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
        std::vector<int64_t> dims1, dims2, dims3;
        int test_index, ndim1, ndim2, ndim3, dtype1, device1, len1, dtype2,
            device2, len2, dtype3, device3, len3, is_causal;
        std::string input_name;

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
        if (!setting.lookupValue("ndim3", ndim3)) {
          std::cerr << "Setting \"ndim3\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        try {
          const libconfig::Setting& dims3_setting = setting.lookup("dims3");
          if (dims3_setting.getLength() != ndim3) {
            std::cerr << " \"dims3\" length is not correct in test index "
                      << test_index << " from " << path << " !\n"
                      << std::endl;
            continue;
          }
          for (int n = 0; n < dims3_setting.getLength(); ++n) {
            dims3.push_back((int64_t) int(dims3_setting[n]));
          }
        } catch (libconfig::SettingNotFoundException& nfex) {
          std::cerr << "Setting \"dims3\" do not exist in test index "
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
        if (!setting.lookupValue("dtype3", dtype3)) {
          std::cerr << "Setting \"dtype3\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("device3", device3)) {
          std::cerr << "Setting \"device3\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("len3", len3)) {
          std::cerr << "Setting \"len3\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("is_causal", is_causal)) {
          std::cerr << "Setting \"is_causal\" do not exist in test index "
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
        Attention_Input tmp(ndim1, dims1, dtype1, device1, nullptr, len1, ndim2,
                            dims2, dtype2, device2, nullptr, len2, ndim3, dims3,
                            dtype3, device3, nullptr, len3, is_causal);
        inputs.push_back(std::move(tmp));
        inputs_name.push_back(input_name);
      }

      for (auto& input : inputs) {
        unsigned int input_nelem1 = 1;
        unsigned int input_nelem2 = 1;
        unsigned int input_nelem3 = 1;

        for (unsigned int j = 0; j < input.ndim1(); j++) {
          input_nelem1 *= input.dims1()[j];
        }
        for (unsigned int j = 0; j < input.ndim2(); j++) {
          input_nelem2 *= input.dims2()[j];
        }
        for (unsigned int j = 0; j < input.ndim3(); j++) {
          input_nelem3 *= input.dims3()[j];
        }
        unsigned int input_len1 = input_nelem1 * elem_size(input.dtype1());
        unsigned int input_len2 = input_nelem2 * elem_size(input.dtype2());
        unsigned int input_len3 = input_nelem3 * elem_size(input.dtype3());
        void* input_data1 = (void*)new char[input_len1];
        void* input_data2 = (void*)new char[input_len2];
        void* input_data3 = (void*)new char[input_len3];
        random_assign(input_data1, input_len1, input.dtype1());
        random_assign(input_data2, input_len2, input.dtype2());
        random_assign(input_data3, input_len3, input.dtype3());
        input.set_data1(input_data1, input_len1);
        input.set_data2(input_data2, input_len2);
        input.set_data3(input_data3, input_len3);
      }
    } catch (const libconfig::SettingNotFoundException& nfex) {
      std::cerr << nfex.getPath() << " do not exist! " << std::endl;
      return (EXIT_FAILURE);
    }
    return (EXIT_SUCCESS);
  }
  using InputType = Attention_Input;
  using UserInterface = InterfaceType;
  // inputs
  std::vector<Attention_Input> attention_inputs;
  std::vector<std::string> attention_name;
  std::map<std::string, int> test_case = {{"attention", 0}};
};
TYPED_TEST_CASE_P(AttentionTest);

TYPED_TEST_P(AttentionTest, TwoTests) {
#ifdef AITISA_API_PYTORCH
  using TorchDataType = typename libtorch_api::DataType;
  using TorchDevice = typename libtorch_api::Device;
  using TorchTensor = typename libtorch_api::Tensor;
#endif
  using time_map_value_attention = std::tuple<double, double>;
  using time_map_attention = std::map<std::string, time_map_value_attention>;
  time_map_attention m;
  auto test = [&m](std::vector<Attention_Input>&& inputs,
                   std::vector<std::string>&& inputs_name,
                   const std::string& test_case_name, int test_case_index) {
    for (int i = 0; i < inputs.size(); i++) {
      auto aitisa_elapsed = std::chrono::duration<double>::zero();
#ifdef AITISA_API_PYTORCH
      auto torch_elapsed = std::chrono::duration<double>::zero();
#endif
      //loop test
      for (int n = 0; n < loop; n++) {
        int64_t aitisa_result_ndim;
        int64_t* aitisa_result_dims = nullptr;
        void* aitisa_result_data = nullptr;
        unsigned int aitisa_result_len;
        AITISA_Tensor aitisa_tensor1, aitisa_tensor2,aitisa_tensor3, aitisa_result;
        AITISA_DataType aitisa_result_dtype;
        AITISA_Device aitisa_result_device;
#ifdef AITISA_API_PYTORCH
        int64_t torch_result_ndim;
        int64_t* torch_result_dims = nullptr;
        void* torch_result_data = nullptr;
        unsigned int torch_result_len;
        TorchTensor torch_tensor1, torch_tensor2,torch_tensor3, torch_result,torch_scores,torch_attn;
        TorchDataType torch_result_dtype;
        TorchDevice torch_result_device(c10::DeviceType::CPU);
#endif
        // aitisa
        AITISA_DataType aitisa_dtype1 = aitisa_int_to_dtype(inputs[i].dtype1());
        AITISA_DataType aitisa_dtype2 = aitisa_int_to_dtype(inputs[i].dtype2());
        AITISA_DataType aitisa_dtype3 = aitisa_int_to_dtype(inputs[i].dtype3());

        AITISA_Device aitisa_device1 = aitisa_int_to_device(0);
        AITISA_Device aitisa_device2 = aitisa_int_to_device(0);
        AITISA_Device aitisa_device3 = aitisa_int_to_device(0);

        aitisa_create(aitisa_dtype1, aitisa_device1, inputs[i].dims1(),
                      inputs[i].ndim1(), (void*)(inputs[i].data1()),
                      inputs[i].len1(), &aitisa_tensor1);
        aitisa_create(aitisa_dtype2, aitisa_device2, inputs[i].dims2(),
                      inputs[i].ndim2(), (void*)(inputs[i].data2()),
                      inputs[i].len2(), &aitisa_tensor2);
        aitisa_create(aitisa_dtype3, aitisa_device3, inputs[i].dims3(),
                      inputs[i].ndim3(), (void*)(inputs[i].data3()),
                      inputs[i].len3(), &aitisa_tensor3);
        auto aitisa_start = std::chrono::steady_clock::now();

        aitisa_attention(aitisa_tensor1, aitisa_tensor2, aitisa_tensor3, inputs[i].is_causal(), &aitisa_result);

        auto aitisa_end = std::chrono::steady_clock::now();
        aitisa_elapsed += aitisa_end - aitisa_start;
        aitisa_resolve(aitisa_result, &aitisa_result_dtype,
                       &aitisa_result_device, &aitisa_result_dims,
                       &aitisa_result_ndim, (void**)&aitisa_result_data,
                       &aitisa_result_len);
#ifdef AITISA_API_PYTORCH
        //torch
        TorchDataType torch_dtype1 =
            libtorch_api::torch_int_to_dtype(inputs[i].dtype1());
        TorchDataType torch_dtype2 =
            libtorch_api::torch_int_to_dtype(inputs[i].dtype2());
        TorchDataType torch_dtype3 =
            libtorch_api::torch_int_to_dtype(inputs[i].dtype3());
        TorchDevice torch_device1 =
            libtorch_api::torch_int_to_device(inputs[i].device1());
        TorchDevice torch_device2 =
            libtorch_api::torch_int_to_device(inputs[i].device2());
        TorchDevice torch_device3 =
            libtorch_api::torch_int_to_device(inputs[i].device3());
        libtorch_api::torch_create(
            torch_dtype1, torch_device1, inputs[i].dims1(), inputs[i].ndim1(),
            inputs[i].data1(), inputs[i].len1(), &torch_tensor1);
        libtorch_api::torch_create(
            torch_dtype2, torch_device2, inputs[i].dims2(), inputs[i].ndim2(),
            inputs[i].data2(), inputs[i].len2(), &torch_tensor2);
        libtorch_api::torch_create(
            torch_dtype3, torch_device3, inputs[i].dims3(), inputs[i].ndim3(),
            inputs[i].data3(), inputs[i].len3(), &torch_tensor3);
        auto torch_start = std::chrono::steady_clock::now();
        //torch attention
        torch_tensor1 = torch::permute(torch_tensor1,{0, 2, 1, 3});
        torch_tensor2 = torch::permute(torch_tensor2,{0, 2, 3, 1});
        torch_tensor3 = torch::permute(torch_tensor3,{0, 2, 1, 3});
        double d = inputs[i].dims1()[3];
        d = sqrt(d);
        torch_scores = torch::matmul(torch_tensor1, torch_tensor2);
        torch_scores = torch::div(torch_scores, d);
        torch_attn = torch::softmax(torch_scores, 3);
        torch_result = torch::matmul(torch_attn, torch_tensor3);
        torch_result = torch::permute(torch_result,{0, 2, 1, 3});

        auto torch_end = std::chrono::steady_clock::now();
        if (!torch_result.is_contiguous()) {
          torch_result = torch_result.contiguous();
        }
        torch_elapsed += torch_end - torch_start;

        libtorch_api::torch_resolve(
            torch_result, &torch_result_dtype, torch_result_device,
            &torch_result_dims, &torch_result_ndim, (void**)&torch_result_data,
            &torch_result_len);
#endif
        // compare
        int64_t tensor_size = 1;

#ifdef AITISA_API_PYTORCH
        ASSERT_EQ(aitisa_result_ndim, torch_result_ndim);
        ASSERT_EQ(0, libtorch_api::torch_device_to_int(torch_result_device));
        ASSERT_EQ(aitisa_dtype_to_int(aitisa_result_dtype),
                  libtorch_api::torch_dtype_to_int(torch_result_dtype));

        for (int64_t j = 0; j < aitisa_result_ndim; j++) {
          tensor_size *= aitisa_result_dims[j];
          ASSERT_EQ(aitisa_result_dims[j], torch_result_dims[j]);
        }
        ASSERT_EQ(aitisa_result_len, torch_result_len);
#endif
        auto* aitisa_data = (float*)aitisa_result_data;
#ifdef AITISA_API_PYTORCH
        auto* torch_data = (float*)torch_result_data;
        for (int64_t j = 0; j < tensor_size; j++) {
          ASSERT_TRUE(abs(aitisa_data[j] - torch_data[j]) < 1e-3);
        }
#endif
        aitisa_tensor1->storage->data = nullptr;
        aitisa_tensor2->storage->data = nullptr;
        aitisa_tensor3->storage->data = nullptr;
        aitisa_destroy(&aitisa_tensor1);
        aitisa_destroy(&aitisa_tensor2);
        aitisa_destroy(&aitisa_tensor3);

        aitisa_destroy(&aitisa_result);
      }
      auto aitisa_time = aitisa_elapsed.count() * 1000 / loop;

      // print result of test
      std::cout << "[ " << test_case_name << " sample" << i << " / "
                << inputs_name[i] << " ] " << std::endl;
      std::cout << "\t[ AITISA ] " << aitisa_time << " ms average for " << loop
                << " loop " << std::endl;

#ifdef AITISA_API_PYTORCH
      auto torch_time = torch_elapsed.count() * 1000 / loop;
      std::cout << "\t[  TORCH  ] " << torch_time << " ms average for " << loop
                << " loop " << std::endl;
      m.insert(
          std::make_pair(test_case_name + " sample " + std::to_string(i),
                         time_map_value_attention(aitisa_time, torch_time)));
#endif
    };
  };
  if (this->attention_inputs.size()) {
    test(std::move(this->attention_inputs), std::move(this->attention_name),
         "attention", this->test_case["attention"]);
    //#ifdef AITISA_API_GENERATE_FIGURE
    //    draw_fig_fun(m, "drop_out");
    //#endif
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(AttentionTest, TwoTests);

#define REGISTER_AttentionTest()                                       \
  class Attention : public Basic {};                                   \
  namespace aitisa_api {                                               \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, AttentionTest, Attention); \
  }

}  // namespace aitisa_api
