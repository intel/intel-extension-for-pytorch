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

#include <string>
#include <unordered_map>

namespace IPEXLogger {

enum class LoggerTypes {
  IPEX_OPS,
  IPEX_MEMORY,
  IPEX_RUNTIME,
  IPEX_SYNGRAPH,
  LOG_MAX // Don't use it
};

} // namespace IPEXLogger