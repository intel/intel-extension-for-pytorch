#pragma once

#include "env_settings.h"

#define IPEX_RECORD_FUNCTION(...)                                          \
  if (torch_ipex::EnvSettings::get_instance().get_settings_profile_op()) { \
    RECORD_FUNCTION(__VA_ARGS__);                                          \
  }