#pragma once

// !USE_GOOGLE_LOG
#include <atomic>
#include <map>
#include <set>
#include <vector>

#include "kineto/ILoggerObserver.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

class LoggerCollector : public ILoggerObserver {
 public:
  LoggerCollector() : buckets_() {}

  void write(const std::string& message, LoggerOutputType ot = ERROR) override {
    if (ot == STAGE || ot == VERBOSE) {
      return;
    }
    buckets_[ot].push_back(message);
  }

  const std::map<LoggerOutputType, std::vector<std::string>>
  extractCollectorMetadata() override {
    return buckets_;
  }

  void reset() override {
    trace_duration_ms = 0;
    event_count = 0;
    destinations.clear();
  }

  void addDevice(const int64_t device) override {
    devices.insert(device);
  }

  void setTraceDurationMS(const int64_t duration) override {
    trace_duration_ms = duration;
  }

  void addEventCount(const int64_t count) override {
    event_count += count;
  }

  void addDestination(const std::string& dest) override {
    if (!dest.empty()) {
      destinations.insert(dest);
    }
  }

  void addMetadata(const std::string& key, const std::string& value) override{};

 protected:
  std::map<LoggerOutputType, std::vector<std::string>> buckets_;

  std::set<int64_t> devices;
  int64_t trace_duration_ms{0};
  std::atomic<uint64_t> event_count{0};
  std::set<std::string> destinations;
};

} // namespace KINETO_NAMESPACE
