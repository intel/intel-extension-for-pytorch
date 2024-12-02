/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include <utils/Settings.h>

#include <vector>
#include "logging_hl.h"
#include "logging_utils.h"
namespace {
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vect) {
  for (auto& elem : vect) {
    os << elem << " ";
  }
  return os;
}

} // namespace

namespace IPEXLogger {
// create loggers (all the log files are created immediately when the module is
// loaded)
static void createModuleLoggers(LoggerTypes) {}

static void createModuleLoggersOnDemandForIPEX(LoggerTypes) {
  hl_logger::LoggerCreateParams default_params;
  default_params.logFileName = "ipex_log.txt";
  default_params.logFileAmount = 5;
  default_params.defaultLoggingLevel = HLLOG_LEVEL_WARN;
  hl_logger::createLoggersOnDemand(
      {LoggerTypes::IPEX_OPS,
       LoggerTypes::IPEX_MEMORY,
       LoggerTypes::IPEX_RUNTIME,
       LoggerTypes::IPEX_SYNGRAPH,
       LoggerTypes::IPEX_TRACE,
       LoggerTypes::IPEX_JIT},
      default_params);
}

static void createModuleLoggersOnDemandForTowl(LoggerTypes) {
  hl_logger::LoggerCreateParams default_params;
  default_params.logFileName = "towl_log.txt";
  default_params.rotateLogfileOnOpen = true;
  default_params.logFileAmount = 5;
  default_params.logFileSize = 3u * 1024u * 1024ul * 1024u;
  default_params.logFileBufferSize = 4u * 1024u * 1024u;
  default_params.defaultLoggingLevel = HLLOG_LEVEL_DEBUG;
  default_params.forceDefaultLoggingLevel = true;
  hl_logger::createLoggersOnDemand({LoggerTypes::IPEX_TOWL}, default_params);
}

static void createModuleLoggersOnDemand(LoggerTypes logger_types) {
  createModuleLoggersOnDemandForIPEX(logger_types);

  if (torch_ipex::xpu::dpcpp::Settings::I().get_towl_logger_enabled()) {
    createModuleLoggersOnDemandForTowl(logger_types);
  }
}

} // namespace IPEXLogger

HLLOG_DEFINE_MODULE_LOGGER(
    IPEX_OPS,
    IPEX_MEMORY,
    IPEX_RUNTIME,
    IPEX_SYNGRAPH,
    IPEX_TRACE,
    IPEX_JIT,
    IPEX_TOWL,
    LOG_MAX)
