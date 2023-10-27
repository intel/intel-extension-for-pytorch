#pragma once

#include <string>

namespace KINETO_NAMESPACE {

class InvariantViolationsLogger {
 public:
  virtual ~InvariantViolationsLogger() = default;
  virtual void logInvariantViolation(
      const std::string& profile_id,
      const std::string& assertion,
      const std::string& error,
      const std::string& group_profile_id) = 0;
  static void registerFactory();
};

} // namespace KINETO_NAMESPACE
