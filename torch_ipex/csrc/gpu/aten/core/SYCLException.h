#pragma once
#include <CL/sycl.hpp>

#include "c10/util/Exception.h"
#include "core/SYCLMacros.h"

#define C10_SYCL_TRY                                       \
  try {

#define C10_SYCL_CATCH_RETHROW(filename, lineno)           \
  } catch (cl::sycl::exception &e) {                       \
    AT_WARN("SYCL Exception: ", e.what(),                  \
            "file = ", filename, "line = ", lineno);       \
    throw;                                                 \
  }

#define C10_SYCL_CATCH_NOTHROW(filename, lineno)           \
  } catch (cl::sycl::exception &e) {                       \
    AT_WARN("SYCL Exception: ", e.what(),                  \
            "file = ", filename, "line = ", lineno);       \
  }

#define __C10_SYCL_CHECK(EXPR, filename, lineno)           \
  do {                                                     \
    int __err;                                             \
    C10_SYCL_TRY                                           \
    __err = EXPR;                                          \
    C10_SYCL_CATCH_RETHROW(filename, lineno)               \
    if (__err != SYCL_SUCCESS) {                           \
      AT_ERROR("SYCL error: %d", __err);                   \
    }                                                      \
  } while (0)

#define __C10_SYCL_CHECK_NORET(EXPR, filename, lineno)     \
  do {                                                     \
    C10_SYCL_TRY                                           \
    (EXPR);                                                \
    C10_SYCL_CATCH_RETHROW(filename, lineno)               \
  } while (0)

#define __C10_SYCL_CHECK_WARN(EXPR, filename, lineno)      \
  do {                                                     \
    C10_SYCL_TRY                                           \
    (EXPR);                                                \
    C10_SYCL_CATCH_NOTHROW(filename, lineno)               \
  } while (0)

#define C10_SYCL_CHECK(EXPR)        __C10_SYCL_CHECK(EXPR, __FILE__, __LINE__)
#define C10_SYCL_CHECK_NORET(EXPR)  __C10_SYCL_CHECK_NORET(EXPR, __FILE__, __LINE__)
#define C10_SYCL_CHECK_WARN(EXPR)   __C10_SYCL_CHECK_WARN(EXPR, __FILE__, __LINE__)

#define C10_SYCL_NOTHROW(...) try { __VA_ARGS__ ; } catch(...) {}

#define C10_SYCL_ASSERT(cond)                              \
  do {                                                     \
    if(!(cond)) {                                          \
      AT_ERROR("assert(%s) failed", #cond);                \
    }                                                      \
  } while(0)


