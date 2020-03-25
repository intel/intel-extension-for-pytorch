#pragma once
#include <CL/sycl.hpp>

#include <c10/util/Exception.h>
#include <core/Macros.h>

#define AT_DPCPP_TRY try {
#define AT_DPCPP_CATCH_RETHROW(filename, lineno) \
  }                                              \
  catch (DPCPP::exception & e) {                 \
    TORCH_WARN(                                  \
        "DPCPP Exception: ",                     \
        e.what(),                                \
        "file = ",                               \
        filename,                                \
        "line = ",                               \
        lineno);                                 \
    throw;                                       \
  }

#define AT_DPCPP_CATCH_NOTHROW(filename, lineno) \
  }                                              \
  catch (DPCPP::exception & e) {                 \
    TORCH_WARN(                                  \
        "DPCPP Exception: ",                     \
        e.what(),                                \
        "file = ",                               \
        filename,                                \
        "line = ",                               \
        lineno);                                 \
  }

#define __AT_DPCPP_CHECK(EXPR, filename, lineno) \
  do {                                           \
    int __err;                                   \
    AT_DPCPP_TRY                                 \
    __err = EXPR;                                \
    AT_DPCPP_CATCH_RETHROW(filename, lineno)     \
    if (__err != DPCPP_SUCCESS) {                \
      TORCH_CHECK(0, "DPCPP error: %d", __err);  \
    }                                            \
  } while (0)

#define __AT_DPCPP_CHECK_NORET(EXPR, filename, lineno) \
  do {                                                 \
    AT_DPCPP_TRY(EXPR);                                \
    AT_DPCPP_CATCH_RETHROW(filename, lineno)           \
  } while (0)

#define __AT_DPCPP_CHECK_WARN(EXPR, filename, lineno) \
  do {                                                \
    AT_DPCPP_TRY(EXPR);                               \
    AT_DPCPP_CATCH_NOTHROW(filename, lineno)          \
  } while (0)

#define AT_DPCPP_CHECK(EXPR) __AT_DPCPP_CHECK(EXPR, __FILE__, __LINE__)
#define AT_DPCPP_CHECK_NORET(EXPR) \
  __AT_DPCPP_CHECK_NORET(EXPR, __FILE__, __LINE__)
#define AT_DPCPP_CHECK_WARN(EXPR) \
  __AT_DPCPP_CHECK_WARN(EXPR, __FILE__, __LINE__)

#define AT_DPCPP_NOTHROW(...) \
  try {                       \
    __VA_ARGS__;              \
  } catch (...) {             \
  }

#define AT_DPCPP_ASSERT(cond)                     \
  do {                                            \
    if (!(cond)) {                                \
      TORCH_CHECK(0, "assert(%s) failed", #cond); \
    }                                             \
  } while (0)
