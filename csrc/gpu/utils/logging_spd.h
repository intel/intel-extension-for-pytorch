/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#pragma once

#include "LogUtils.h"
#include "logging_utils.h"

enum class LoggerTypes {
  IPEX_OPS,
  IPEX_MEMORY,
  IPEX_RUNTIME,
  IPEX_SYNGRAPH,
  LOG_MAX // Don't use it
};

#include <iostream>
#define VA_ARGS_TO_FMT_AND_ARGS_TRACE(STRING_MOD, FMT, ...)                 \
  if (static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_OPS) {      \
    IPEX_TRACE_LOG("OPS", "", FMT, __VA_ARGS__);                            \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_MEMORY) {   \
    IPEX_TRACE_LOG("MEMORY", "", FMT, __VA_ARGS__);                         \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_RUNTIME) {  \
    IPEX_TRACE_LOG("RUNTIME", "", FMT, __VA_ARGS__);                        \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_SYNGRAPH) { \
    IPEX_TRACE_LOG("SYNGRAPH", "", FMT, __VA_ARGS__);                       \
  }

#define VA_ARGS_TO_FMT_AND_ARGS_DEBUG(STRING_MOD, FMT, ...)                 \
  if (static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_OPS) {      \
    IPEX_DEBUG_LOG("OPS", "", FMT, __VA_ARGS__);                            \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_MEMORY) {   \
    IPEX_DEBUG_LOG("MEMORY", "", FMT, __VA_ARGS__);                         \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_RUNTIME) {  \
    IPEX_DEBUG_LOG("RUNTIME", "", FMT, __VA_ARGS__);                        \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_SYNGRAPH) { \
    IPEX_DEBUG_LOG("SYNGRAPH", "", FMT, __VA_ARGS__);                       \
  }

#define VA_ARGS_TO_FMT_AND_ARGS_INFO(STRING_MOD, FMT, ...)                  \
  if (static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_OPS) {      \
    IPEX_INFO_LOG("OPS", "", FMT, __VA_ARGS__);                             \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_MEMORY) {   \
    IPEX_INFO_LOG("MEMORY", "", FMT, __VA_ARGS__);                          \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_RUNTIME) {  \
    IPEX_INFO_LOG("RUNTIME", "", FMT, __VA_ARGS__);                         \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_SYNGRAPH) { \
    IPEX_INFO_LOG("SYNGRAPH", "", FMT, __VA_ARGS__);                        \
  }

#define VA_ARGS_TO_FMT_AND_ARGS_WARN(STRING_MOD, FMT, ...)                  \
  if (static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_OPS) {      \
    IPEX_WARN_LOG("OPS", "", FMT, __VA_ARGS__);                             \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_MEMORY) {   \
    IPEX_WARN_LOG("MEMORY", "", FMT, __VA_ARGS__);                          \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_RUNTIME) {  \
    IPEX_WARN_LOG("RUNTIME", "", FMT, __VA_ARGS__);                         \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_SYNGRAPH) { \
    IPEX_WARN_LOG("SYNGRAPH", "", FMT, __VA_ARGS__);                        \
  }

#define VA_ARGS_TO_FMT_AND_ARGS_ERROR(STRING_MOD, FMT, ...)                 \
  if (static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_OPS) {      \
    IPEX_ERR_LOG("OPS", "", FMT, __VA_ARGS__);                              \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_MEMORY) {   \
    IPEX_ERR_LOG("MEMORY", "", FMT, __VA_ARGS__);                           \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_RUNTIME) {  \
    IPEX_ERR_LOG("RUNTIME", "", FMT, __VA_ARGS__);                          \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_SYNGRAPH) { \
    IPEX_ERR_LOG("SYNGRAPH", "", FMT, __VA_ARGS__);                         \
  }

#define VA_ARGS_TO_FMT_AND_ARGS_FATAL(STRING_MOD, FMT, ...)                 \
  if (static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_OPS) {      \
    IPEX_FATAL_LOG("OPS", "", FMT, __VA_ARGS__);                            \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_MEMORY) {   \
    IPEX_FATAL_LOG("MEMORY", "", FMT, __VA_ARGS__);                         \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_RUNTIME) {  \
    IPEX_FATAL_LOG("RUNTIME", "", FMT, __VA_ARGS__);                        \
  } else if (                                                               \
      static_cast<LoggerTypes>(STRING_MOD) == LoggerTypes::IPEX_SYNGRAPH) { \
    IPEX_FATAL_LOG("SYNGRAPH", "", FMT, __VA_ARGS__);                       \
  }

#define IPEX_MOD_TRACE(MOD, ...) \
  VA_ARGS_TO_FMT_AND_ARGS_TRACE(static_cast<LoggerTypes>(MOD), __VA_ARGS__);

#define IPEX_MOD_DEBUG(MOD, ...) \
  VA_ARGS_TO_FMT_AND_ARGS_DEBUG(static_cast<LoggerTypes>(MOD), __VA_ARGS__);

#define IPEX_MOD_INFO(MOD, ...) \
  VA_ARGS_TO_FMT_AND_ARGS_INFO(static_cast<LoggerTypes>(MOD), __VA_ARGS__);

#define IPEX_MOD_WARN(MOD, ...) \
  VA_ARGS_TO_FMT_AND_ARGS_WARN(static_cast<LoggerTypes>(MOD), __VA_ARGS__);

#define IPEX_MOD_ERROR(MOD, ...) \
  VA_ARGS_TO_FMT_AND_ARGS_ERROR(static_cast<LoggerTypes>(MOD), __VA_ARGS__);

#define IPEX_MOD_FATAL(MOD, ...) \
  VA_ARGS_TO_FMT_AND_ARGS_FATAL(static_cast<LoggerTypes>(MOD), __VA_ARGS__);
