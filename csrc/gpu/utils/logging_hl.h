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

#include <fmt/ostream.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <hl_logger/hllog.hpp>

#include <ATen/core/jit_type.h>
#include <ATen/core/operator_name.h>
#include <ATen/core/symbol.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include "logging_utils.h"

#define CREATE_OSTREAM_FORMATTER(type) \
  template <>                          \
  struct fmt::formatter<type> : ostream_formatter {};

CREATE_OSTREAM_FORMATTER(at::ScalarType);
CREATE_OSTREAM_FORMATTER(std::vector<int64_t>);
CREATE_OSTREAM_FORMATTER(std::vector<const char*>);
CREATE_OSTREAM_FORMATTER(std::vector<std::string>);
CREATE_OSTREAM_FORMATTER(c10::Device);
CREATE_OSTREAM_FORMATTER(c10::OperatorName);
CREATE_OSTREAM_FORMATTER(c10::Symbol);
CREATE_OSTREAM_FORMATTER(c10::Type);

#define BRACED_PARAM(p) "{}"
#define FORMAT_AND_MSG(...) \
  HLLOG_APPLY(HLLOG_EMPTY, BRACED_PARAM, ##__VA_ARGS__), ##__VA_ARGS__

#define HLLOG_ENUM_TYPE_NAME IPEXLogger::LoggerTypes
HLLOG_DECLARE_MODULE_LOGGER()

#define IPEX_MOD_TRACE(MOD, ...) HLLOG_TRACE(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_DEBUG(MOD, ...) HLLOG_DEBUG(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_INFO(MOD, ...) HLLOG_INFO(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_WARN(MOD, ...) HLLOG_WARN(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_ERROR(MOD, ...) HLLOG_ERR(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define IPEX_MOD_FATAL(MOD, ...) \
  HLLOG_CRITICAL(MOD, FORMAT_AND_MSG(__VA_ARGS__));
