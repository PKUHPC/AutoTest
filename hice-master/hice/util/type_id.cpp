#include "hice/util/type_id.h"

using std::string;

namespace hice {

TypeId TypeId::create() {
  static std::atomic<TypeId::underlying_type> 
      counter(DataType::id<_HighestPreallocatedTypeId>().underlying_id());
  const TypeId::underlying_type new_value = ++counter;
  if (new_value == std::numeric_limits<TypeId::underlying_type>::max()) {
    throw std::logic_error(
        "Ran out of available type ids. If you need more than 2^16 KNOWN_TYPEs, " 
        "we need to increase TypeId to use more than 16 bit.");
  }
  return TypeId(new_value);
}

namespace detail {
const TypeData _type_data_instance_uninitialized_ = TypeData(TypeId::uninitialized(), 
    0, "nullptr (uninitialized)", nullptr, nullptr, nullptr, nullptr, nullptr);
} // namespace detail

template<>
const detail::TypeData* DataType::type_data_instance<detail::_Uninitialized>() noexcept {
  return &detail::_type_data_instance_uninitialized_;
}

HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(0, uint8_t)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(1, int8_t)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(2, uint16_t)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(3, int16_t)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(4, uint32_t)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(5, int32_t)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(6, uint64_t)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(7, int64_t)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(8, float)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(9, double)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(10, std::complex<float>)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(11, std::complex<double>)
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(12, bool)
// 13 = undefined type id
HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(14, _HighestPreallocatedTypeId)


} // namespace hice 
