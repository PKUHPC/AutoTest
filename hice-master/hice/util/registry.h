// This file is based on hice\util\IdWrapper.h from PyTorch. 
// PyTorch is BSD-style licensed, as found in its LICENSE file.
// From https://github.com/pytorch/pytorch.
// And it's modified for HICE's usage.

#pragma once
/**
 * Simple registry implementation that uses static variables to
 * register object creators during program initialization time.
 */

// NB: This Registry works poorly when you have other namespaces.
// Make all macro invocations from inside the at namespace.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "hice/core/macros.h"

namespace hice {

template <typename KeyType>
inline std::string key_str_repr(const KeyType& /*key*/) {
  return "[key type printing not supported]";
}

template <>
inline std::string key_str_repr(const std::string& key) {
  return key;
}

enum RegistryPriority {
  REGISTRY_FALLBACK = 1,
  REGISTRY_DEFAULT = 2,
  REGISTRY_PREFERRED = 3,
};

/**
 * @brief A template class that allows one to register classes by keys.
 *
 * The keys are usually a std::string specifying the name, but can be anything
 * that can be used in a std::map.
 *
 * You should most likely not use the Registry class explicitly, but use the
 * helper macros below to declare specific registries as well as registering
 * objects.
 */
template <class KeyType, class ObjectPtrType, class... Args>
class Registry {
 public:
  typedef std::function<ObjectPtrType(Args...)> Creator;

  Registry() : registry_(), priority_(), terminate_(true) {}

  void register_creator(
      const KeyType& key,
      Creator creator,
      const RegistryPriority priority = REGISTRY_DEFAULT) {
    std::lock_guard<std::mutex> lock(register_mutex_);
    // The if statement below is essentially the same as the following line:
    // CHECK_EQ(registry_.count(key), 0) << "Key " << key
    //                                   << " registered twice.";
    // However, CHECK_EQ depends on google logging, and since registration is
    // carried out at static initialization time, we do not want to have an
    // explicit dependency on glog's initialization function.
    if (registry_.count(key) != 0) {
      auto cur_priority = priority_[key];
      if (priority > cur_priority) {
        std::string warn_msg =
            "Overwriting already registered item for key " + key_str_repr(key);
        fprintf(stderr, "%s\n", warn_msg.c_str());
        registry_[key] = creator;
        priority_[key] = priority;
      } else if (priority == cur_priority) {
        std::string err_msg =
            "Key already registered with the same priority: " + key_str_repr(key);
        fprintf(stderr, "%s\n", err_msg.c_str());
        if (terminate_) {
          std::exit(1);
        } else {
          throw std::runtime_error(err_msg);
        }
      } else {
        std::string warn_msg =
            "Higher priority item already registered, skipping registration of " +
            key_str_repr(key);
        fprintf(stderr, "%s\n", warn_msg.c_str());
      }
    } else {
      registry_[key] = creator;
      priority_[key] = priority;
    }
  }

  void register_creator(
      const KeyType& key,
      Creator creator,
      const std::string& help_msg,
      const RegistryPriority priority = REGISTRY_DEFAULT) {
    register_creator(key, creator, priority);
    help_message_[key] = help_msg;
  }

  inline bool has(const KeyType& key) {
    return (registry_.count(key) != 0);
  }

  ObjectPtrType create(const KeyType& key, Args... args) {
    if (registry_.count(key) == 0) {
      // Returns nullptr if the key is not registered.
      return nullptr;
    }
    return registry_[key](args...);
  }

  /**
   * Returns the keys currently registered as a std::vector.
   */
  std::vector<KeyType> keys() const {
    std::vector<KeyType> keys;
    for (const auto& it : registry_) {
      keys.push_back(it.first);
    }
    return keys;
  }

  inline const std::unordered_map<KeyType, std::string>& help_message() const {
    return help_message_;
  }

  const char* help_message(const KeyType& key) const {
    auto it = help_message_.find(key);
    if (it == help_message_.end()) {
      return nullptr;
    }
    return it->second.c_str();
  }

  // Used for testing, if terminate is unset, Registry throws instead of
  // calling std::exit
  void set_terminate(bool terminate) {
    terminate_ = terminate;
  }

 private:
  std::unordered_map<KeyType, Creator> registry_;
  std::unordered_map<KeyType, RegistryPriority> priority_;
  bool terminate_;
  std::unordered_map<KeyType, std::string> help_message_;
  std::mutex register_mutex_;

  HICE_DISABLE_COPY_AND_ASSIGN(Registry);
};

template <class KeyType, class ObjectPtrType, class... Args>
class Registrar {
 public:
  explicit Registrar(
      const KeyType& key,
      Registry<KeyType, ObjectPtrType, Args...>* registry,
      typename Registry<KeyType, ObjectPtrType, Args...>::Creator creator,
      const std::string& help_msg = "") {
    registry->register_creator(key, creator, help_msg);
  }

  explicit Registrar(
      const KeyType& key,
      const RegistryPriority priority,
      Registry<KeyType, ObjectPtrType, Args...>* registry,
      typename Registry<KeyType, ObjectPtrType, Args...>::Creator creator,
      const std::string& help_msg = "") {
    registry->register_creator(key, creator, help_msg, priority);
  }

  template <class DerivedType>
  static ObjectPtrType default_creator(Args... args) {
    return ObjectPtrType(new DerivedType(args...));
  }
};

/**
 * HICE_DECLARE_TYPED_REGISTRY is a macro that expands to a function
 * declaration, as well as creating a convenient typename for its corresponding
 * registerer.
 */
// Note on HICE_IMPORT and HICE_EXPORT below: we need to explicitly mark DECLARE
// as import and DEFINE as export, because these registry macros will be used
// in downstream shared libraries as well, and one cannot use *_API - the API
// macro will be defined on a per-shared-library basis. Semantically, when one
// declares a typed registry it is always going to be IMPORT, and when one
// defines a registry (which should happen ONLY ONCE and ONLY IN SOURCE FILE),
// the instantiation unit is always going to be exported.
//
// The only unique condition is when in the same file one does DECLARE and
// DEFINE - in Windows compilers, this generates a warning that dllimport and
// dllexport are mixed, but the warning is fine and linker will be properly
// exporting the symbol. Same thing happens in the gflags flag declaration and
// definition caes.
#define HICE_DECLARE_TYPED_REGISTRY(                                         \
    RegistryName, KeyType, ObjectType, PtrType, ...)                         \
    HICE_IMPORT hice::Registry<KeyType, PtrType<ObjectType>, ##__VA_ARGS__>* \
  RegistryName();                                                            \
  typedef hice::Registrar<KeyType, PtrType<ObjectType>, ##__VA_ARGS__>       \
      Registrar##RegistryName

#define HICE_DEFINE_TYPED_REGISTRY(                                           \
    RegistryName, KeyType, ObjectType, PtrType, ...)                          \
    HICE_EXPORT hice::Registry<KeyType, PtrType<ObjectType>, ##__VA_ARGS__>*  \
  RegistryName() {                                                            \
    static hice::Registry<KeyType, PtrType<ObjectType>, ##__VA_ARGS__>*       \
        registry = new hice::                                                 \
            Registry<KeyType, PtrType<ObjectType>, ##__VA_ARGS__>();          \
    return registry;                                                          \
  }

// Note(Yangqing): The __VA_ARGS__ below allows one to specify a templated
// creator with comma in its templated arguments.
#define HICE_REGISTER_TYPED_CREATOR(RegistryName, key, ...)                  \
  static Registrar##RegistryName HICE_ANONYMOUS_VARIABLE(g_##RegistryName)(   \
      key, RegistryName(), ##__VA_ARGS__);

#define HICE_REGISTER_TYPED_CREATOR_WITH_PRIORITY(                           \
    RegistryName, key, priority, ...)                                        \
  static Registrar##RegistryName HICE_ANONYMOUS_VARIABLE(g_##RegistryName)(   \
      key, priority, RegistryName(), ##__VA_ARGS__);

#define HICE_REGISTER_TYPED_CLASS(RegistryName, key, ...)                    \
  static Registrar##RegistryName HICE_ANONYMOUS_VARIABLE(g_##RegistryName)(   \
      key,                                                                   \
      RegistryName(),                                                        \
      Registrar##RegistryName::default_creator<__VA_ARGS__>,                  \
      hice::demangle_type<__VA_ARGS__>());

#define HICE_REGISTER_TYPED_CLASS_WITH_PRIORITY(                             \
    RegistryName, key, priority, ...)                                       \
  static Registrar##RegistryName HICE_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                                  \
      priority,                                                             \
      RegistryName(),                                                       \
      Registrar##RegistryName::default_creator<__VA_ARGS__>,                \
      hice::demangle_type<__VA_ARGS__>());

// HICE_DECLARE_REGISTRY and HICE_DEFINE_REGISTRY are hard-wired to use
// std::string as the key type, because that is the most commonly used cases.
#define HICE_DECLARE_REGISTRY(RegistryName, ObjectType, ...) \
  HICE_DECLARE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define HICE_DEFINE_REGISTRY(RegistryName, ObjectType, ...) \
  HICE_DEFINE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define HICE_DECLARE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  HICE_DECLARE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

#define HICE_DEFINE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  HICE_DEFINE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

// HICE_REGISTER_CREATOR and HICE_REGISTER_CLASS are hard-wired to use std::string
// as the key type, because that is the most commonly used cases.
#define HICE_REGISTER_CREATOR(RegistryName, key, ...) \
  HICE_REGISTER_TYPED_CREATOR(RegistryName, #key, __VA_ARGS__)

#define HICE_REGISTER_CREATOR_WITH_PRIORITY(RegistryName, key, priority, ...) \
  HICE_REGISTER_TYPED_CREATOR_WITH_PRIORITY(                                  \
      RegistryName, #key, priority, __VA_ARGS__)

#define HICE_REGISTER_CLASS(RegistryName, key, ...) \
  HICE_REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

#define HICE_REGISTER_CLASS_WITH_PRIORITY(RegistryName, key, priority, ...) \
  HICE_REGISTER_TYPED_CLASS_WITH_PRIORITY(                                  \
      RegistryName, #key, priority, __VA_ARGS__)

} // namespace hice