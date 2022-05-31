#include "env_settings.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace torch_ipex {

EnvSettings::EnvSettings() {
  initialize_all_settings();
}

EnvSettings& EnvSettings::get_instance() {
  static EnvSettings _instance;

  return _instance;
}

void EnvSettings::initialize_all_settings() {
  auto envar = std::getenv("IPEX_PROFILE_OP");
  if (envar) {
    if (strcmp(envar, "1") == 0) {
      m_b_profile_op_ = true;
    }
  }
}

bool EnvSettings::get_settings_profile_op() {
  return m_b_profile_op_;
}

void EnvSettings::set_settings_profile_op(bool b_enable) {
  m_b_profile_op_ = b_enable;
}

} // namespace torch_ipex
