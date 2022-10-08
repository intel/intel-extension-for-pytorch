#pragma once

#include <iostream>
#include "include/Macros.h"

#define IPEX_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;         \
  classname& operator=(const classname&) = delete

// Host side print utils
template <typename T>
void ipex_host_print(T& t) {
  std::cout << t;
}

template <typename T, typename... Ts>
void ipex_host_print(T& t, Ts&&... args) {
  std::cout << t;
  ipex_host_print(std::forward<Ts>(args)...);
}

#define IPEX_H_PRINT(...)         \
  {                               \
    ipex_host_print(__VA_ARGS__); \
    std::cout << std::endl;       \
  }

#define IPEX_IDENTIFY_1(IDENTIFY_CONCAT, WHAT, X, ...) (WHAT(X))
#define IPEX_IDENTIFY_2(IDENTIFY_CONCAT, WHAT, X, ...) \
  IDENTIFY_CONCAT(                                     \
      (WHAT(X)), IPEX_IDENTIFY_1(IDENTIFY_CONCAT, WHAT, __VA_ARGS__))
#define IPEX_IDENTIFY_3(IDENTIFY_CONCAT, WHAT, X, ...) \
  IDENTIFY_CONCAT(                                     \
      (WHAT(X)), IPEX_IDENTIFY_2(IDENTIFY_CONCAT, WHAT, __VA_ARGS__))
#define IPEX_IDENTIFY_4(IDENTIFY_CONCAT, WHAT, X, ...) \
  IDENTIFY_CONCAT(                                     \
      (WHAT(X)), IPEX_IDENTIFY_3(IDENTIFY_CONCAT, WHAT, __VA_ARGS__))
#define IPEX_IDENTIFY_5(IDENTIFY_CONCAT, WHAT, X, ...) \
  IDENTIFY_CONCAT(                                     \
      (WHAT(X)), IPEX_IDENTIFY_4(IDENTIFY_CONCAT, WHAT, __VA_ARGS__))
#define IPEX_IDENTIFY_6(IDENTIFY_CONCAT, WHAT, X, ...) \
  IDENTIFY_CONCAT(                                     \
      (WHAT(X)), IPEX_IDENTIFY_5(IDENTIFY_CONCAT, WHAT, __VA_ARGS__))
#define IPEX_IDENTIFY_7(IDENTIFY_CONCAT, WHAT, X, ...) \
  IDENTIFY_CONCAT(                                     \
      (WHAT(X)), IPEX_IDENTIFY_6(IDENTIFY_CONCAT, WHAT, __VA_ARGS__))
#define IPEX_IDENTIFY_8(IDENTIFY_CONCAT, WHAT, X, ...) \
  IDENTIFY_CONCAT(                                     \
      (WHAT(X)), IPEX_IDENTIFY_7(IDENTIFY_CONCAT, WHAT, __VA_ARGS__))
#define IPEX_IDENTIFY_FLAGCAT(X, Y) X##Y
#define IPEX_IDENTIFY_(N, IDENTIFY_CONCAT, WHAT, X, ...) \
  IPEX_IDENTIFY_FLAGCAT(IPEX_IDENTIFY_, N)               \
  (IDENTIFY_CONCAT, WHAT, X, __VA_ARGS__)

#define IPEX_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0
#define IPEX_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define IPEX_NARG_(...) IPEX_ARG_N(__VA_ARGS__)
#define IPEX_NARG(...) IPEX_NARG_(__VA_ARGS__, IPEX_RSEQ_N())

#define IPEX_IDENTIFY_CONCAT_ANY(X, Y) (X) || (Y)
#define IPEX_IDENTIFY_CONCAT_ALL(X, Y) (X) && (Y)

#define IPEX_ANY(WHAT, ...) \
  (IPEX_IDENTIFY_(          \
      IPEX_NARG(__VA_ARGS__), IPEX_IDENTIFY_CONCAT_ANY, WHAT, __VA_ARGS__))
#define IPEX_ALL(WHAT, ...) \
  (IPEX_IDENTIFY_(          \
      IPEX_NARG(__VA_ARGS__), IPEX_IDENTIFY_CONCAT_ALL, WHAT, __VA_ARGS__))
