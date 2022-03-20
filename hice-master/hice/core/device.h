// This file is based on hice\core\Device.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's slightly modified for HICE's usage. 

#pragma once

#include <ostream>


#include <hice/core/macros.h>

namespace hice {

using DeviceIndex = int16_t;

enum class DeviceType: int16_t {
  CPU = 0,
  CUDA = 1,
  NumDeviceTypes = 2,
};

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr int kNumDeviceTypes = static_cast<int>(DeviceType::NumDeviceTypes);

bool is_valid_device_type(DeviceType d);

std::string to_string(DeviceType d, bool lower_case = false);

std::ostream& operator<<(std::ostream& stream, DeviceType type);

struct HICE_API Device {


  Device()=default;

  Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {
    validate();
  }

  Device(const std::string& device_string);


  bool operator==(const Device& other) const noexcept {
      return this->type_ == other.type_ && this->index_ == other.index_;
  }

  bool operator!=(const Device& other) const noexcept { 
    return !(*this == other);
  }

  DeviceType type() const noexcept { return type_; }

  DeviceIndex index() const noexcept { return index_; }

  void set_index(DeviceIndex index) { index_ = index; }

  bool has_index() const noexcept { return index_ != -1; }

  bool is_cpu() const noexcept { return type_ == DeviceType::CPU; }

  bool is_cuda() const noexcept { return type_ == DeviceType::CUDA; }

 private:
  void validate();
  DeviceType type_;
  DeviceIndex index_;
};

HICE_API std::ostream& operator<<(std::ostream& os, const Device& device);

} // namespace hice

namespace std {

template <>
struct hash<hice::Device> {
  size_t operator()(hice::Device d) const noexcept {
    // Are you here because this static assert failed?  Make sure you ensure
    // that the bitmasking code below is updated accordingly!
    static_assert(sizeof(hice::DeviceType) == 2, "DeviceType is not 16-bit");
    static_assert(sizeof(hice::DeviceIndex) == 2, "DeviceIndex is not 16-bit");
    // Note [Hazard when concatenating signed integers]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // We must first convert to a same-sized unsigned type, before promoting to
    // the result type, to prevent sign extension when any of the values is -1.
    // If sign extension occurs, you'll clobber all of the values in the MSB
    // half of the resulting integer.
    //
    // Technically, by C/C++ integer promotion rules, we only need one of the
    // uint32_t casts to the result type, but we put in both for explicitness's sake.
    uint32_t bits =
        static_cast<uint32_t>(static_cast<uint16_t>(d.type())) << 16
      | static_cast<uint32_t>(static_cast<uint16_t>(d.index()));
    return std::hash<uint32_t>{}(bits);
  }
};

template <> struct hash<hice::DeviceType> {
  std::size_t operator()(hice::DeviceType k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};

} // namespace std