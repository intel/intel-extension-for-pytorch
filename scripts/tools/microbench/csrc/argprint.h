#pragma once

#include <torch/extension.h>
#include <iostream>
#include <string>
using namespace at;

template <typename T>
void print_(T item) {
  std::cout << "unknown";
}

#define PRINT_TYPE_WRAP(X) \
  template <>              \
  void print_<X>(X item)

PRINT_TYPE_WRAP(int64_t) {
  std::cout << item;
}
PRINT_TYPE_WRAP(double) {
  std::cout << item;
}
PRINT_TYPE_WRAP(bool) {
  std::cout << item;
}
PRINT_TYPE_WRAP(Scalar) {
  std::cout << item;
}
PRINT_TYPE_WRAP(ScalarType) {
  std::cout << item;
}
PRINT_TYPE_WRAP(IntArrayRef) {
  std::cout << item;
}

#define PRINT_TYPE_WRAP_OPTIONAL(X) \
  PRINT_TYPE_WRAP(X) {              \
    if (item.has_value()) {         \
      std::cout << item.value();    \
    } else {                        \
      std::cout << "undef";         \
    }                               \
  }

PRINT_TYPE_WRAP_OPTIONAL(c10::optional<int64_t>)
PRINT_TYPE_WRAP_OPTIONAL(c10::optional<double>)
PRINT_TYPE_WRAP_OPTIONAL(c10::optional<bool>)
PRINT_TYPE_WRAP_OPTIONAL(c10::optional<Scalar>)
PRINT_TYPE_WRAP_OPTIONAL(c10::optional<ScalarType>)
PRINT_TYPE_WRAP_OPTIONAL(c10::optional<IntArrayRef>)

PRINT_TYPE_WRAP(Tensor) {
  if (item.defined()) {
    std::cout << item.dtype() << item.sizes() << item.suggest_memory_format();
  } else {
    std::cout << "undef";
  }
}

PRINT_TYPE_WRAP(c10::optional<Tensor>) {
  if (item.has_value() && item.value().defined()) {
    std::cout << item.value().dtype() << item.value().sizes()
              << item.value().suggest_memory_format();
  } else {
    std::cout << "undef";
  }
}

template <>
void print_<std::array<bool, 3>>(std::array<bool, 3> item) {
  std::cout << "[" << item[0] << "," << item[1] << "," << item[2] << "]";
}

template <>
void print_<c10::optional<std::array<bool, 3>>>(
    c10::optional<std::array<bool, 3>> item) {
  if (item.has_value()) {
    auto v = item.value();
    std::cout << "[" << v[0] << "," << v[1] << "," << v[2] << "]";
  } else {
    std::cout << "undef";
  }
}

template <typename T>
void argprint(T item, std::string argname, std::string argtype) {
  std::cout << argname << "(" << argtype << "):";
  print_(item);
  std::cout << "; ";
}
