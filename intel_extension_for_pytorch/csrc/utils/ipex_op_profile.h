#pragma once

#include "env_settings.h"

#define RECORD_FUNCTION_WITH_SCOPE_AND_SWITCH(scope, fn, inputs, switch, ...) \
  at::RecordFunction guard(scope);                                            \
  if (switch) {                                                               \
    if (guard.isActive()) {                                                   \
      if (guard.needsInputs()) {                                              \
        guard.before(fn, inputs, ##__VA_ARGS__);                              \
      } else {                                                                \
        guard.before(fn, ##__VA_ARGS__);                                      \
      }                                                                       \
    }                                                                         \
  }

// this MACRO is forked from pytorch RECORD_FUNCTION(fn, inputs, ...)
#define RECORD_FUNCTION_WITH_SWTICH(fn, inputs, ...)                     \
  bool __b_is_turn_on =                                                  \
      torch_ipex::EnvSettings::get_instance().get_settings_profile_op(); \
  RECORD_FUNCTION_WITH_SCOPE_AND_SWITCH(                                 \
      at::RecordScope::FUNCTION, fn, inputs, __b_is_turn_on, ##__VA_ARGS__)

#define IPEX_RECORD_FUNCTION(...) RECORD_FUNCTION_WITH_SWTICH(__VA_ARGS__)
