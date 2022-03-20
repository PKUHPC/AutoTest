#pragma once

#include <complex>

namespace hice {

namespace ext {

#ifdef __cpp_lib_logical_traits
template <bool B>
using bool_constant = std::bool_constant<B>;
template <class B>
using negation = std::negation<B>;
#else
// Implementation taken from http://en.cppreference.com/w/cpp/types/integral_constant
template <bool B>
using bool_constant = std::integral_constant<bool, B>;
// Implementation taken from http://en.cppreference.com/w/cpp/types/negation
template<class B>
struct negation : bool_constant<!bool(B::value)> { };
#endif

#ifdef __cpp_lib_transformation_trait_aliases
template<bool B, class T = void> using enable_if_t = std::enable_if_t<B, T>;
template<class T> using decay_t = std::decay_t<T>;
template<class T> using remove_reference_t = std::remove_reference_t<T>;
template<class T> using remove_const_t = std::remove_const_t<T>;
template<class T> using remove_cv_t = std::remove_cv_t<T>;
#else
template<bool B, class T = void> using enable_if_t = typename std::enable_if<B, T>::type;
template<class T> using decay_t = typename std::decay<T>::type;
template<class T> using remove_reference_t = typename std::remove_reference<T>::type;
template<class T> using remove_const_t = typename std::remove_const<T>::type;
template<class T> using remove_cv_t = typename std::remove_cv<T>::type;
#endif


template <typename T>
struct is_complex : public std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

template <typename T>
struct scalar_value_type {
  using type = T;
};
template <typename T>
struct scalar_value_type<std::complex<T>> {
  using type = T;
};

template <typename TEnum>
constexpr typename std::enable_if<std::is_enum<TEnum>::value, 
                                  typename std::underlying_type<TEnum>::type>::type 
to_number(const TEnum value) {
      return static_cast<typename std::underlying_type<TEnum>::type>(value);
}

} // namespace ext

} // namespace hice