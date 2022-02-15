#pragma once

#include <string>

namespace torch_ipex {

class EnvSettings {
 private:
  EnvSettings();
  bool m_b_profile_op_ = false;

 public:
  static EnvSettings& get_instance();
  void initialize_all_settings();

 public:
  bool get_settings_profile_op();
};

} // namespace torch_ipex
