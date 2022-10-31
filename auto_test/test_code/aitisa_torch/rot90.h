#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include "auto_test/basic.h"
#include "auto_test/sample.h"
#include "hice/core/tensor_printer.h"

extern "C" {
#include "src/new_ops1/rot90.h"
}

namespace aitisa_api {

namespace {

class Rot90_Input : public Unary_Input {
 public:
  Rot90_Input() = default;

  Rot90_Input(int64_t ndim, std::vector<int64_t> dims, int dtype, int device,
              void* data, unsigned int len, int k)
      : Unary_Input(ndim, std::move(dims), dtype, device, data, len), k_(k) {}
  Rot90_Input(Rot90_Input&& input) noexcept
      : Unary_Input(input.ndim(), input.dims(), input.dtype(), input.device(),
                    input.data(), input.len()),
        k_(input.k()) {
    input.to_nullptr();
  }
  ~Rot90_Input() override = default;

  Rot90_Input& operator=(Rot90_Input& right) {
    auto& left = (Unary_Input&)(*this);
    left = (Unary_Input&)right;
    this->k_ = right.k();
  }
  int k() const { return k_; }

 private:
  int k_ = 0;
};
}  // namespace

template <typename InterfaceType>
class Rot90Test : public ::testing::Test {
 public:
  Rot90Test() { fetch_test_data("rot90", rot90_inputs, rot90_name); }
  ~Rot90Test() override = default;
  static void aitisa_kernel(const AITISA_Tensor input, const int k,
                            AITISA_Tensor* output) {
    aitisa_rot90(input, k, output);
  }
  int fetch_test_data(const char* path, std::vector<Rot90_Input>& inputs,
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
        int test_index, ndim, dtype, device, len, k;
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
        if (!setting.lookupValue("k", k)) {
          std::cerr << "Setting \"k\" do not exist in test index " << test_index
                    << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        if (!setting.lookupValue("input_name", input_name)) {
          std::cerr << "Setting \"input_name\" do not exist in test index "
                    << test_index << " from " << path << " !\n"
                    << std::endl;
          continue;
        }
        Rot90_Input tmp(ndim, dims, dtype, device, nullptr, len, k);
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
  using InputType = Rot90_Input;
  using UserInterface = InterfaceType;
  // inputs
  std::vector<Rot90_Input> rot90_inputs;
  std::vector<std::string> rot90_name;
  std::map<std::string, int> test_case = {{"rot90", 0}};
};
TYPED_TEST_CASE_P(Rot90Test);

TYPED_TEST_P(Rot90Test, TwoTests) {
#ifdef AITISA_API_PYTORCH
  using TorchDataType = typename libtorch_api::DataType;
  using TorchDevice = typename libtorch_api::Device;
  using TorchTensor = typename libtorch_api::Tensor;
#endif
  using time_map_value_rot90 = std::tuple<double, double>;
  using time_map_rot90 = std::map<std::string, time_map_value_rot90>;
  time_map_rot90 m;
  auto test = [&m](std::vector<Rot90_Input>&& inputs,
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
        float* aitisa_result_data = nullptr;
        unsigned int aitisa_result_len;
        AITISA_Tensor aitisa_tensor, aitisa_result;
        AITISA_DataType aitisa_result_dtype;
        AITISA_Device aitisa_result_device;
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

        aitisa_rot90(aitisa_tensor, inputs[i].k(), &aitisa_result);

        auto aitisa_end = std::chrono::steady_clock::now();
        aitisa_elapsed += aitisa_end - aitisa_start;
        aitisa_resolve(aitisa_result, &aitisa_result_dtype,
                       &aitisa_result_device, &aitisa_result_dims,
                       &aitisa_result_ndim, (void**)&aitisa_result_data,
                       &aitisa_result_len);
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
        if (inputs[i].ndim() == 3) {
          torch_result = torch::rot90(torch_tensor, inputs[i].k(), {1, 2});
        } else if (inputs[i].ndim() == 4) {
          torch_result = torch::rot90(torch_tensor, inputs[i].k(), {2, 3});
        }

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
          ASSERT_FLOAT_EQ(aitisa_data[j], torch_data[j]);
        }

#endif
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
      m.insert(std::make_pair(test_case_name + " sample " + std::to_string(i),
                              time_map_value_rot90(aitisa_time, torch_time)));
#endif
    };
  };
  if (this->rot90_inputs.size()) {
    test(std::move(this->rot90_inputs), std::move(this->rot90_name), "rot90",
         this->test_case["rot90"]);
    //#ifdef AITISA_API_GENERATE_FIGURE
    //    draw_fig_fun(m, "drop_out");
    //#endif
  } else
    FAIL() << "No input test case.";
}
REGISTER_TYPED_TEST_CASE_P(Rot90Test, TwoTests);

#define REGISTER_ROT90()                                       \
  class Rot90 : public Basic {};                               \
  namespace aitisa_api {                                       \
  INSTANTIATE_TYPED_TEST_CASE_P(aitisa_api, Rot90Test, Rot90); \
  }

}  // namespace aitisa_api
