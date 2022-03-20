// This file is based on c10\core\TensorOptions.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage.

#pragma once

#include "hice/util/type_id.h"
#include "hice/util/traits.h"
#include "hice/util/types.h"
#include "hice/core/layout.h"
#include "hice/core/device.h"
#include "hice/core/scalar_type.h"

namespace hice {

struct HICE_API TensorOptions {
  TensorOptions() {}

  template<typename T,
           typename = ext::enable_if_t<std::is_same<ext::decay_t<T>, Device>::value>>
  TensorOptions(T&& device) : TensorOptions() {
    this->set_device(std::forward<T>(device));
  }

  template <typename... Args,
            typename = ext::enable_if_t<std::is_constructible<Device, Args&&...>::value>>
  TensorOptions(Args&&... args)
      : TensorOptions(Device(std::forward<Args>(args)...)) {}

  TensorOptions(LayoutType layout_type) : TensorOptions() {
    this->set_layout(layout_type);
  }

  TensorOptions(DataType dtype) : TensorOptions() {
    this->set_dtype(dtype);
  }

  TensorOptions(ScalarType scalar_type) : TensorOptions() {
    this->set_dtype(scalar_type);
  }

  bool operator==(const TensorOptions& other) const noexcept {
    return
        has_dtype_ == other.has_dtype_ &&
        has_device_ == other.has_device_ &&
        has_layout_ == other.has_layout_ &&
        (!has_dtype_ || dtype_ == other.dtype_) &&
        (!has_device_ || device_ == other.device_) &&
        (!has_layout_ || layout_ == other.layout_);
  }

  bool operator!=(const TensorOptions& other) const noexcept {
    return !(*this == other);
  }

  DeviceType device_type() const noexcept {
    return has_device_ ? device_.type() : DeviceType::CPU;
  }

  Device device() const noexcept {
    return has_device_ ? device_ : Device(DeviceType::CPU);
  }

  bool has_device() const noexcept {
    return has_device_;
  }

  LayoutType layout_type() const noexcept {
    return has_layout_ ? layout_.type() : kDense;
  }

  Layout layout() const noexcept {
    return has_layout_ ? layout_ : Layout(kDense);
  }

  bool has_layout() const noexcept {
    return has_layout_;
  }

  ScalarType scalar_type() const noexcept {
    return has_dtype_ ? DataTypeToScalarType(dtype_) : ScalarType::Float;
  }

  DataType data_type() const noexcept {
    return has_dtype_ ? dtype_ : DataType::make<float>();
  }

  bool has_dtype() const noexcept {
    return has_dtype_;
  }

  TensorOptions device(hice::optional<Device> device) const noexcept {
    TensorOptions r = *this;
    r.set_device(device);
    return r;
  }

  template<typename ... Args>
  TensorOptions device(Args&&... args) const noexcept {
    return device(hice::optional<Device>(in_place, std::forward<Args>(args)...));
  }

  TensorOptions layout(hice::optional<Layout> layout) const noexcept {
    TensorOptions r = *this;
    r.set_layout(layout);
    return r;
  }

  template<typename ... Args>
  TensorOptions layout(Args&&... args) const noexcept {
    return layout(hice::optional<Layout>(in_place, std::forward<Args>(args)...));
  }

  TensorOptions dtype(hice::optional<DataType> dtype) const noexcept {
    TensorOptions r = *this;
    r.set_dtype(dtype);
    return r;
  }

  TensorOptions dtype(hice::optional<ScalarType> scalar_type) const noexcept {
    TensorOptions r = *this;
    r.set_dtype(scalar_type);
    return r;
  }

  template <typename T>
  TensorOptions& dtype() {
    dtype_ = DataType::make<T>();
    has_dtype_ = true;
    return *this;
  }

 private:

  void set_device(hice::optional<Device> device) & noexcept {
    if (device) {
      device_ = *device;
      has_device_ = true;
    } else {
      has_device_ = false;
    }
  }

  void set_layout(hice::optional<Layout> layout) & noexcept {
    if (layout) {
      layout_ = *layout;
      has_layout_ = true;
    } else {
      has_layout_ = false;
    }
  }

  void set_dtype(hice::optional<DataType> dtype) & noexcept {
    if (dtype) {
      dtype_ = *dtype;
      has_dtype_ = true;
    } else {
      has_dtype_ = false;
    }
  }

  void set_dtype(hice::optional<ScalarType> scalar_type) & noexcept {
    if (scalar_type) {
      dtype_ = ScalarTypeToDataType(*scalar_type);
      has_dtype_ = true;
    } else {
      has_dtype_ = false;
    }
  }

  Device device_ = DeviceType::CPU;
  DataType dtype_ = DataType::make<float>();
  Layout layout_ = kDense;

  bool has_device_ = false;
  bool has_dtype_ = false;
  bool has_layout_ = false;
};

inline TensorOptions dtype(DataType dtype) {
  return TensorOptions().dtype(dtype);
}

inline TensorOptions dtype(ScalarType scalar_type) {
  return TensorOptions().dtype(ScalarTypeToDataType(scalar_type));
}

template <typename T>
inline TensorOptions dtype() {
  return dtype(DataType::make<T>());
}

inline TensorOptions layout(Layout layout) {
  return TensorOptions().layout(layout);
}

inline TensorOptions device(Device device) {
  return TensorOptions().device(std::move(device));
}

inline std::ostream& operator<<(std::ostream& stream,
                                const TensorOptions& options) {
  return stream << "TensorOptions(dtype=" << options.data_type()
                << ", device=" << options.device()
                << ", layout=" << options.layout() << ")";
}

} // namespace hice
