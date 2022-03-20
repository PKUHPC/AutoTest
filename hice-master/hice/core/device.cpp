#include <algorithm>
#include <array>
#include <exception>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "hice/util/loguru.h"
#include "hice/core/device.h"

namespace hice {

// NB: Per the C++ standard (e.g.,
// https://stackoverflow.com/questions/18195312/what-happens-if-you-static-cast-invalid-value-to-enum-class)
// as long as you cast from the same underlying type, it is always valid to cast
// into an enum class (even if the value would be invalid by the enum.)  Thus,
// the caller is allowed to cast a possibly invalid int16_t to DeviceType and
// then pass it to this function.  (I considered making this function take an
// int16_t directly, but that just seemed weird.)
bool is_valid_device_type(DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
    case DeviceType::CUDA:
      return true;
    default:
      return false;
  }
}

std::string to_string(DeviceType d, bool lower_case) {
  switch (d) {
    // I considered instead using ctype::tolower to lower-case the strings
    // on the fly, but this seemed a bit much.
    case DeviceType::CPU:
      return lower_case ? "cpu" : "CPU";
    case DeviceType::CUDA:
      return lower_case ? "cuda" : "CUDA";
    default:
      HICE_LOG(ERROR) << "Unknown device: " << static_cast<int16_t>(d);
      return "";
  }
}


std::ostream& operator<<(std::ostream& stream, DeviceType type) {
  stream << to_string(type, /* lower case */ true);
  return stream;
}

namespace {
DeviceType parse_type(const std::string& device_string) {
  static const std::array<std::pair<std::string, DeviceType>, 2> types = {{
      {"cpu", DeviceType::CPU},
      {"cuda", DeviceType::CUDA},
  }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [device_string](const std::pair<std::string, DeviceType>& p) {
        return p.first == device_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  HICE_LOG(ERROR) << "Expected one of cpu, cuda device type at start of device string: " 
             << device_string;
}

} // namespace

void Device::validate() {
  HICE_CHECK(index_ == -1 || index_ >= 0)
      << "Device index must be -1 or non-negative, got " 
      << index_;
  HICE_CHECK(!is_cpu() || index_ <= 0)
      << "CPU device index must be -1 or zero, got "
      << index_;
}

// `std::regex` is still in a very incomplete state in GCC 4.8.x,
// so we have to do our own parsing, like peasants.
// https://stackoverflow.com/questions/12530406/is-gcc-4-8-or-earlier-buggy-about-regular-expressions
//
// Replace with the following code once we shed our GCC skin:
//
// static const std::regex regex(
//     "(cuda|cpu)|(cuda|cpu):([0-9]+)|([0-9]+)",
//     std::regex_constants::basic);
// std::smatch match;
// const bool ok = std::regex_match(device_string, match, regex);
// HICE_CHECK(ok, "Invalid device string: '", device_string, "'");
// if (match[1].matched) {
//   type_ = parse_type_from_string(match[1].str());
// } else {
//   if (match[2].matched) {
//     type_ = parse_type_from_string(match[1].str());
//   } else {
//     type_ = Type::kCUDA;
//   }
//   ASSERT(match[3].matched);
//   index_ = std::stoi(match[3].str());
// }
Device::Device(const std::string& device_string) : Device(DeviceType::CPU) {
  HICE_CHECK(!device_string.empty()) << "Device string must not be empty";
  int index = device_string.find(":");
  if (index == std::string::npos) {
    type_ = parse_type(device_string);
  } else {
    std::string s;
    s = device_string.substr(0, index);
    HICE_CHECK(!s.empty()) << "Device string must not be empty";
    type_ = parse_type(s);

    std::string device_index = device_string.substr(index + 1);
    try {
      index_ = std::stoi(device_index);
    } catch (const std::exception &) {
      HICE_LOG(ERROR) << "Could not parse device index '" << device_index 
                 << "' in device string '" << device_string << "'";
    }
    HICE_CHECK(index_ >= 0) << "Device index must be non-negative, got " << index_;
  }
  validate();
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.type();
  if (device.has_index()) {
    stream << ":" << device.index();
  }
  return stream;
}

} // namespace hice 
