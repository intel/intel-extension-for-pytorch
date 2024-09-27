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

#if defined(BUILD_SYNGRAPH)
#include "logging_hl.h"
// ----------------------------------- TRACE ---------------------------------
#define IPEX_OPS_TRACE IPEX_MOD_TRACE(IPEX_OPS, __func__)
#define IPEX_MEMORY_TRACE IPEX_MOD_TRACE(IPEX_MEMORY, __func__)
#define IPEX_RUNTIME_TRACE IPEX_MOD_TRACE(IPEX_RUNTIME, __func__)
#define IPEX_SYNGRAPH_TRACE IPEX_MOD_TRACE(IPEX_SYNGRAPH, __func__)

// ----------------------------------- DEBUG ---------------------------------
#define IPEX_OPS_DEBUG(...) IPEX_MOD_DEBUG(IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_DEBUG(...) IPEX_MOD_DEBUG(IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_DEBUG(...) IPEX_MOD_DEBUG(IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_DEBUG(...) IPEX_MOD_DEBUG(IPEX_SYNGRAPH, __VA_ARGS__)

// ----------------------------------- INFO ---------------------------------
#define IPEX_OPS_INFO(...) IPEX_MOD_INFO(IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_INFO(...) IPEX_MOD_INFO(IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_INFO(...) IPEX_MOD_INFO(IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_INFO(...) IPEX_MOD_INFO(IPEX_SYNGRAPH, __VA_ARGS__)

// ----------------------------------- WARN ----------------------------------
#define IPEX_OPS_WARN(...) IPEX_MOD_WARN(IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_WARN(...) IPEX_MOD_WARN(IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_WARN(...) IPEX_MOD_WARN(IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_WARN(...) IPEX_MOD_WARN(IPEX_SYNGRAPH, __VA_ARGS__)

// ----------------------------------- ERROR ---------------------------------
#define IPEX_OPS_ERROR(...) IPEX_MOD_ERROR(IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_ERROR(...) IPEX_MOD_ERROR(IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_ERROR(...) IPEX_MOD_ERROR(IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_ERROR(...) IPEX_MOD_ERROR(IPEX_SYNGRAPH, __VA_ARGS__)

// --------------------------------- FATAL --------------------------------
#define IPEX_OPS_FATAL(...) IPEX_MOD_FATAL(IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_FATAL(...) IPEX_MOD_FATAL(IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_FATAL(...) IPEX_MOD_FATAL(IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_FATAL(...) IPEX_MOD_FATAL(IPEX_SYNGRAPH, __VA_ARGS__)

// ------------------------------- NOT IMPLEMENTED ---------------------------
#define IPEX_MOD_NOT_IMPLEMENTED(MOD) \
  IPEX_MOD_WARN(                      \
      MOD,                            \
      "Function ",                    \
      __func__,                       \
      " not implemented! @",          \
      __FILE__,                       \
      ":",                            \
      __LINE__)
#define IPEX_OPS_NOT_IMPLEMENTED IPEX_MOD_NOT_IMPLEMENTED(IPEX_OPS)
#define IPEX_MEMORY_NOT_IMPLEMENTED IPEX_MOD_NOT_IMPLEMENTED(IPEX_MEMORY)
#define IPEX_RUNTIME_NOT_IMPLEMENTED IPEX_MOD_NOT_IMPLEMENTED(IPEX_RUNTIME)
#define IPEX_SYNGRAPH_NOT_IMPLEMENTED IPEX_MOD_NOT_IMPLEMENTED(IPEX_SYNGRAPH)

// ----------------------------------- ASSERT --------------------------------
#define VA_ARGS(...) , ##__VA_ARGS__
#define IPEX_MOD_ASSERT(MOD, condition, ...) \
  {                                          \
    if (!(condition)) {                      \
      IPEX_MOD_FATAL(                        \
          MOD,                               \
          "Asertion failed [",               \
          #condition,                        \
          "] " VA_ARGS(__VA_ARGS__),         \
          " @",                              \
          __FILE__,                          \
          ":",                               \
          __LINE__)                          \
    }                                        \
  }

#define IPEX_OPS_ASSERT(condition, ...) \
  IPEX_MOD_ASSERT(IPEX_OPS, condition, __VA_ARGS__)
#define IPEX_MEMORY_ASSERT(condition, ...) \
  IPEX_MOD_ASSERT(IPEX_MEMORY, condition, __VA_ARGS__)
#define IPEX_RUNTIME_ASSERT(condition, ...) \
  IPEX_MOD_ASSERT(IPEX_RUNTIME, condition, __VA_ARGS__)
#define IPEX_SYNGRAPH_ASSERT(condition, ...) \
  IPEX_MOD_ASSERT(IPEX_SYNGRAPH, condition, __VA_ARGS__)

#else
#include "logging_spd.h"
// ----------------------------------- TRACE ---------------------------------
#define IPEX_OPS_TRACE IPEX_MOD_TRACE(LoggerTypes::IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_TRACE IPEX_MOD_TRACE(LoggerTypes::IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_TRACE \
  IPEX_MOD_TRACE(LoggerTypes::IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_TRACE \
  IPEX_MOD_TRACE(LoggerTypes::IPEX_SYNGRAPH, __VA_ARGS__)

// ----------------------------------- DEBUG ---------------------------------
#define IPEX_OPS_DEBUG(...) IPEX_MOD_DEBUG(LoggerTypes::IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_DEBUG(...) \
  IPEX_MOD_DEBUG(LoggerTypes::IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_DEBUG(...) \
  IPEX_MOD_DEBUG(LoggerTypes::IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_DEBUG(...) \
  IPEX_MOD_DEBUG(LoggerTypes::IPEX_SYNGRAPH, __VA_ARGS__)

// ----------------------------------- INFO ---------------------------------
#define IPEX_OPS_INFO(...)               \
  std::cout << "Inside IPEX_OPS_INFO\n"; \
  IPEX_MOD_INFO(LoggerTypes::IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_INFO(...) \
  IPEX_MOD_INFO(LoggerTypes::IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_INFO(...) \
  IPEX_MOD_INFO(LoggerTypes::IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_INFO(...) \
  IPEX_MOD_INFO(LoggerTypes::IPEX_SYNGRAPH, __VA_ARGS__)

// ----------------------------------- WARN ----------------------------------
#define IPEX_OPS_WARN(...) IPEX_MOD_WARN(LoggerTypes::IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_WARN(...) \
  IPEX_MOD_WARN(LoggerTypes::IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_WARN(...) \
  IPEX_MOD_WARN(LoggerTypes::IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_WARN(...) \
  IPEX_MOD_WARN(LoggerTypes::IPEX_SYNGRAPH, __VA_ARGS__)

// ----------------------------------- ERROR ---------------------------------
#define IPEX_OPS_ERROR(...) IPEX_MOD_ERROR(LoggerTypes::IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_ERROR(...) \
  IPEX_MOD_ERROR(LoggerTypes::IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_ERROR(...) \
  IPEX_MOD_ERROR(LoggerTypes::IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_ERROR(...) \
  IPEX_MOD_ERROR(LoggerTypes::IPEX_SYNGRAPH, __VA_ARGS__)

// --------------------------------- FATAL --------------------------------
#define IPEX_OPS_FATAL(...) IPEX_MOD_FATAL(LoggerTypes::IPEX_OPS, __VA_ARGS__)
#define IPEX_MEMORY_FATAL(...) \
  IPEX_MOD_FATAL(LoggerTypes::IPEX_MEMORY, __VA_ARGS__)
#define IPEX_RUNTIME_FATAL(...) \
  IPEX_MOD_FATAL(LoggerTypes::IPEX_RUNTIME, __VA_ARGS__)
#define IPEX_SYNGRAPH_FATAL(...) \
  IPEX_MOD_FATAL(LoggerTypes::IPEX_SYNGRAPH, __VA_ARGS__)

// ------------------------------- NOT IMPLEMENTED ---------------------------
#define IPEX_MOD_NOT_IMPLEMENTED(MOD) \
  IPEX_MOD_WARN(                      \
      MOD,                            \
      "Function ",                    \
      __func__,                       \
      " not implemented! @",          \
      __FILE__,                       \
      ":",                            \
      __LINE__)
#define IPEX_OPS_NOT_IMPLEMENTED IPEX_MOD_NOT_IMPLEMENTED(LoggerTypes::IPEX_OPS)
#define IPEX_MEMORY_NOT_IMPLEMENTED \
  IPEX_MOD_NOT_IMPLEMENTED(LoggerTypes::IPEX_MEMORY)
#define IPEX_RUNTIME_NOT_IMPLEMENTED \
  IPEX_MOD_NOT_IMPLEMENTED(LoggerTypes::IPEX_RUNTIME)
#define IPEX_SYNGRAPH_NOT_IMPLEMENTED \
  IPEX_MOD_NOT_IMPLEMENTED(LoggerTypes::IPEX_SYNGRAPH)

// ----------------------------------- ASSERT --------------------------------
#define VA_ARGS(...) , ##__VA_ARGS__
#define IPEX_MOD_ASSERT(MOD, condition, ...) \
  {                                          \
    if (!(condition)) {                      \
      IPEX_MOD_FATAL(                        \
          MOD,                               \
          "Asertion failed [",               \
          #condition,                        \
          "] " VA_ARGS(__VA_ARGS__),         \
          " @",                              \
          __FILE__,                          \
          ":",                               \
          __LINE__)                          \
    }                                        \
  }

#define IPEX_OPS_ASSERT(condition, ...) \
  IPEX_MOD_ASSERT(LoggerTypes::IPEX_OPS, condition, __VA_ARGS__)
#define IPEX_MEMORY_ASSERT(condition, ...) \
  IPEX_MOD_ASSERT(LoggerTypes::IPEX_MEMORY, condition, __VA_ARGS__)
#define IPEX_RUNTIME_ASSERT(condition, ...) \
  IPEX_MOD_ASSERT(LoggerTypes::IPEX_RUNTIME, condition, __VA_ARGS__)
#define IPEX_SYNGRAPH_ASSERT(condition, ...) \
  IPEX_MOD_ASSERT(LoggerTypes::IPEX_SYNGRAPH, condition, __VA_ARGS__)

#endif
