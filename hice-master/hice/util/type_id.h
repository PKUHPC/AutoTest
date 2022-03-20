// This file is based on c10\util\typeid.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage. 

#pragma once

#include <atomic>
#include <complex>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <type_traits>

#include "hice/core/macros.h"
#include "hice/util/traits.h"
#include "hice/util/loguru.h"
#include "hice/util/id_wrapper.h"
#include "hice/util/type_demangle.h"

namespace hice {

/**
 * A type id is a unique id for a given C++ type.
 * You need to register your types using KNOWN_TYPE(MyType) to be able to
 * use TypeId with custom types. This is for example used to store the
 * dtype of tensors.
 */
class HICE_API TypeId final
    : public IdWrapper<TypeId, uint16_t> {
 public:
  static TypeId create();

  friend std::ostream& operator<<(std::ostream& stream, TypeId typeId);

  friend bool operator<(TypeId lhs, TypeId rhs);

  // 0 is uint8_t (due to ScalarType BC constraint)
  static constexpr TypeId uninitialized() {
    return TypeId(13);
  }

  /**
   * Returns the unique id for the given type T. The id is unique for the type T
   * in the sense that for any two different types, their ids are different; for
   * the same type T, the id remains the same over different calls of the
   * function. However, this is not guaranteed over different runs, as the id
   * is generated during run-time. Do NOT serialize the id for storage.
   */
  template <typename T>
  static TypeId get();

 private:
  constexpr explicit TypeId(uint16_t id) : IdWrapper(id) {}

  friend class DataType;
};

// Allow usage in std::map / std::set
// TODO Disallow this and rather use std::unordered_map/set everywhere
inline bool operator<(TypeId lhs, TypeId rhs) {
  return lhs.underlying_id() < rhs.underlying_id();
}

inline std::ostream& operator<<(std::ostream& stream, TypeId type_id) {
  return stream << type_id.underlying_id();
}

namespace detail {

struct TypeData final {
  using New = void*();
  using PlacementNew = void(void*, size_t);
  using Copy = void(const void*, void*, size_t);
  using PlacementDelete = void(void*, size_t);
  using Delete = void(void*);

  TypeData() = delete;

  constexpr TypeData(
    TypeId id,
    size_t size,
    const char* name,
    New* new_fn,
    PlacementNew* placement_new_fn,
    Copy* copy_fn,
    PlacementDelete* placement_delete_fn,
    Delete* delete_fn) noexcept
     : id_(id), size_(size), name_(name), 
       new_fn_(new_fn), placement_new_fn_(placement_new_fn), copy_fn_(copy_fn),
       placement_delete_fn_(placement_delete_fn), delete_fn_(delete_fn) {}

  TypeId id_;
  size_t size_;
  const char* name_;
  New* new_fn_;
  PlacementNew* placement_new_fn_;
  Copy* copy_fn_;
  PlacementDelete* placement_delete_fn_;
  Delete* delete_fn_;
};

/**
 * Placement new function for the type.
 */
template <typename T>
inline void _placement_new_fn(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (size_t i = 0; i < n; ++i) {
    new (typed_ptr + i) T;
  }
}

template <typename T>
inline void _placement_new_fn_not_default(void* /*ptr*/, size_t /*n*/) {
  HICE_LOG(ERROR) << "Type is not default-constructible";
}

template<typename T,
         ext::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeData::PlacementNew* _get_placement_new_fn() {
  return
    (std::is_fundamental<T>::value || std::is_pointer<T>::value)
    ? nullptr
    : &_placement_new_fn<T>;
}

template<typename T,
         ext::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeData::PlacementNew* _get_placement_new_fn() {
  static_assert(!std::is_fundamental<T>::value && !std::is_pointer<T>::value, 
                "this should have got the other SFINAE case");
  return &_placement_new_fn_not_default<T>;
}

template <typename T>
inline void* _new_fn() {
  return new T;
}

template <typename T>
inline void* _new_fn_not_default() {
  HICE_LOG(ERROR) << "Type is not default-constructible";
}

template<typename T,
         ext::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeData::New* _get_new_fn() {
  return &_new_fn<T>;
}

template <typename T,
          ext::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeData::New* _get_new_fn() {
  return &_new_fn_not_default<T>;
}

/**
 * Typed copy function for classes.
 */
template <typename T>
inline void _copy_fn(const void* src, void* dst, size_t n) {
  const T* typed_src = static_cast<const T*>(src);
  T* typed_dst = static_cast<T*>(dst);
  for (size_t i = 0; i < n; ++i) {
    typed_dst[i] = typed_src[i];
  }
}

template <typename T>
inline void _copy_fn_not_allowed(const void* /*src*/, void* /*dst*/, size_t /*n*/) {
  HICE_LOG(ERROR) << "Type does not allow assignment.";
}

template<typename T,
         ext::enable_if_t<std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr TypeData::Copy* _get_copy_fn() {
  return
    (std::is_fundamental<T>::value || std::is_pointer<T>::value)
    ? nullptr
    : &_copy_fn<T>;
}

template<typename T,
         ext::enable_if_t<!std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr TypeData::Copy* _get_copy_fn() {
  static_assert(!std::is_fundamental<T>::value && !std::is_pointer<T>::value, 
                "this should have got the other SFINAE case");
  return &_copy_fn_not_allowed<T>;
}

/**
 * Destructor for non-fundamental types.
 */
template <typename T>
inline void _placement_delete_fn(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (size_t i = 0; i < n; ++i) {
    typed_ptr[i].~T();
  }
}

template <typename T>
inline constexpr TypeData::PlacementDelete* _get_placement_delete_fn() {
  return
    (std::is_fundamental<T>::value || std::is_pointer<T>::value)
    ? nullptr
    : &_placement_delete_fn<T>;
}

template <typename T>
inline void _delete_fn(void* ptr) {
  T* typed_ptr = static_cast<T*>(ptr);
  delete typed_ptr;
}

template<class T>
inline constexpr TypeData::Delete* _get_delete_fn() noexcept {
  return &_delete_fn<T>;
}


#ifdef __GXX_RTTI
template <class T>
const char* _type_name(const char* literalName) noexcept {
  std::ignore = literalName; // suppress unused warning
  static const std::string name = demangle(typeid(T).name());
  return name.c_str();
}
#else
template <class T>
constexpr const char* _type_name(const char* literalName) noexcept {
  return literalName;
}
#endif

template<class T>
inline TypeData _make_type_data_instance(const char* type_name) {
  return {
    TypeId::get<T>(),
    sizeof(T),
    type_name,
    _get_new_fn<T>(),
    _get_placement_new_fn<T>(),
    _get_copy_fn<T>(),
    _get_placement_delete_fn<T>(),
    _get_delete_fn<T>(),
  };
}

class _Uninitialized final {};

} // namespace detail


class HICE_API DataType {
 public:
  using New = detail::TypeData::New;
  using PlacementNew = detail::TypeData::PlacementNew;
  using Copy = detail::TypeData::Copy;
  using PlacementDelete = detail::TypeData::PlacementDelete;
  using Delete = detail::TypeData::Delete;

  DataType() noexcept;

  constexpr DataType(const DataType& src) noexcept = default;

  DataType& operator=(const DataType& src) noexcept = default;

  constexpr DataType(DataType&& rhs) noexcept = default;

 private:
  // DataType can only be created by Make, making sure that we do not
  // create incorrectly mixed up TypeMeta objects.
  explicit constexpr DataType(const detail::TypeData* data) noexcept : data_(data) {}

 public:

  constexpr TypeId id() const noexcept {
    return data_->id_;
  }

  constexpr const char* name() const noexcept {
    return data_->name_;
  }

  constexpr size_t size() const noexcept {
    return data_->size_;
  }

  constexpr New* new_fn() const noexcept {
    return data_->new_fn_;
  }

  constexpr PlacementNew* placement_new_fn() const noexcept {
    return data_->placement_new_fn_;
  }

  constexpr Copy* copy_fn() const noexcept {
    return data_->copy_fn_;
  }

  constexpr PlacementDelete* placement_delete_fn() const noexcept {
    return data_->placement_delete_fn_;
  }

  constexpr Delete* delete_fn() const noexcept {
    return data_->delete_fn_;
  }

  friend bool operator==(const DataType& lhs, const DataType& rhs) noexcept;

  template <typename T>
  constexpr bool match() const noexcept {
    return (*this == make<T>());
  }

  // Below are static functions that can be called by passing a specific type.

  template <class T>
  static TypeId id() noexcept {
    return TypeId::get<T>();
  }

  template <class T>
  static const char* name() noexcept {
    return make<T>().name();
  }

  template <class T>
  static constexpr size_t size() noexcept {
    return sizeof(T);
  }

  template <typename T>
  static DataType make() {
    // The instance pointed to is declared here, but defined in a .cpp file.
    // We need to silence the compiler warning about using an undefined
    // variable template. '-Wpragmas' and '-Wunknown-warning-option' has to be
    // disabled for compilers that don't know '-Wundefined-var-template' and
    // would error at our attempt to disable it.
#ifndef _MSC_VER
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpragmas"
#  pragma GCC diagnostic ignored "-Wunknown-warning-option"
#  pragma GCC diagnostic ignored "-Wundefined-var-template"
#endif
    return DataType(type_data_instance<T>());
#ifndef _MSC_VER
#  pragma GCC diagnostic pop
#endif
  }

 private:
  template <class T>
  HICE_API static const detail::TypeData* type_data_instance() noexcept;

  const detail::TypeData *data_;
};

template <>
HICE_API const detail::TypeData*
DataType::type_data_instance<detail::_Uninitialized>() noexcept;

inline DataType::DataType() noexcept : data_(type_data_instance<detail::_Uninitialized>()) {}

inline bool operator==(const DataType& lhs, const DataType& rhs) noexcept {
  return (lhs.data_ == rhs.data_);
}

inline bool operator!=(const DataType& lhs, const DataType& rhs) noexcept {
  return !operator==(lhs, rhs);
}

inline std::ostream& operator<<(std::ostream& stream, DataType dtype) {
  return stream << dtype.name();
}

/**
 * Register unique id for a type so it can be used in DataType context, e.g. be
 * used as a type for Tensor elements.
 *
 * HICE_KNOWN_TYPE does explicit instantiation of TypeId::Get<T>
 * template function and thus needs to be put in a single translation unit
 * (.cpp file) for a given type T. Other translation units that use type T 
 * as a element type of HICE::Tensor need to depend on the translation unit 
 * that contains HICE_KNOWN_TYPE declaration via regular linkage dependencies.
 */

// Implementation note: in MSVC, we will need to prepend the HICE_API
// keyword in order to get things compiled properly. in Linux, gcc seems to
// create attribute ignored error for explicit template instantiations, see
//   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0537r0.html
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51930
// and as a result, we define these two macros slightly differently.
#if defined(_MSC_VER) || defined(__clang__)
#define HICE_EXPORT_IF_NOT_GCC HICE_EXPORT
#else
#define HICE_EXPORT_IF_NOT_GCC
#endif

#define _HICE_KNOWN_TYPE_DEFINE_TYPEDATA_INSTANCE(T, Counter)         \
  namespace detail {                                                  \
  const TypeData HICE_MACRO_CONCAT(_type_data_instance_, Counter) =   \
      _make_type_data_instance<T>(_type_name<T>(#T));                 \
  }                                                                   \
  template <>                                                         \
  HICE_EXPORT_IF_NOT_GCC const detail::TypeData*                      \
  TypeData::type_data_instance<T>() noexcept {                        \
    return &HICE_MACRO_CONCAT(detail::_type_data_instance_, Counter); \
  }

#define HICE_KNOWN_TYPE(T)                          \
  template <>                                       \
  HICE_EXPORT_IF_NOT_GCC TypeId TypeId::get<T>() {  \
    static const TypeId type_id = TypeId::create(); \
    return type_id;                                 \
  }                                                 \
  _HICE_KNOWN_TYPE_DEFINE_TYPEDATA_INSTANCE(T, __COUNTER__)

/**
 * HICE_DECLARE_PREALLOCATED_KNOWN_TYPE is used
 * to preallocate ids for types that are queried very often so that they
 * can be resolved at compile time. Please use KNOWN_TYPE() instead
 * for your own types to allocate dynamic ids for them.
 */
#ifdef _MSC_VER
#define HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(preallocated_id, T) \
  template <>                                                    \
  HICE_EXPORT inline TypeId TypeId::Get<T>() {                   \
    return TypeId(preallocated_id);                              \
  }                                                              \
  namespace detail {                                             \
  HICE_API extern const TypeData HICE_MACRO_CONCAT(              \
      _type_data_instance_preallocated_, preallocated_id);       \
  }

#define HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(preallocated_id, T)          \
  namespace detail {                                                     \
  HICE_EXPORT const TypeData HICE_MACRO_CONCAT(                          \
      _type_data_instance_preallocated_,                                 \
      preallocated_id) = _make_type_data_instance<T>(_type_name<T>(#T)); \
  }                                                                      \
  template <>                                                            \
  HICE_EXPORT const detail::TypeData*                                    \
  DataType::type_data_instance<T>() noexcept {                           \
    return &HICE_MACRO_CONCAT(detail::_type_data_instance_preallocated_, \
                              preallocated_id);                          \
  }
#else  // _MSC_VER
#define HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(preallocated_id, T)         \
  template <>                                                            \
  HICE_EXPORT inline TypeId TypeId::get<T>() {                           \
    return TypeId(preallocated_id);                                      \
  }                                                                      \
  namespace detail {                                                     \
  HICE_API extern const TypeData HICE_MACRO_CONCAT(                      \
      _type_data_instance_preallocated_, preallocated_id);               \
  }                                                                      \
  template <>                                                            \
  HICE_EXPORT inline const detail::TypeData*                             \
  DataType::type_data_instance<T>() noexcept {                           \
    return &HICE_MACRO_CONCAT(detail::_type_data_instance_preallocated_, \
                              preallocated_id);                          \
  }

#define HICE_DEFINE_PREALLOCATED_KNOWN_TYPE(preallocated_id, T)       \
  namespace detail {                                                  \
  const TypeData HICE_MACRO_CONCAT(_type_data_instance_preallocated_, \
                                   preallocated_id) =                 \
      _make_type_data_instance<T>(_type_name<T>(#T));                 \
  }

#endif

// Note: we have preallocated the numbers so they line up exactly
// with ScalarType's numbering.  All other numbers do not matter.
struct _HighestPreallocatedTypeId final {};

HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(0, uint8_t)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(1, int8_t)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(2, uint16_t)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(3, int16_t)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(4, uint32_t)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(5, int32_t)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(6, uint64_t)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(7, int64_t)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(8, float)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(9, double)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(10, std::complex<float>)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(11, std::complex<double>)
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(12, bool)
// 13 = undefined type id
HICE_DECLARE_PREALLOCATED_KNOWN_TYPE(14, _HighestPreallocatedTypeId)

} // namespace hice 
