// !USE_GOOGLE_LOG
#include <time.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include <profiler/Logger.h>
#include <profiler/include/kineto/ILoggerObserver.h>
#include <profiler/include/kineto/ThreadUtil.h>

namespace KINETO_NAMESPACE {

std::atomic_int Logger::severityLevel_{VERBOSE};
std::atomic_int Logger::verboseLogLevel_{-1};
std::atomic<uint64_t> Logger::verboseLogModules_{~0ull};

Logger::Logger(int severity, int line, const char* filePath, int errnum)
    : buf_(),
      out_(LIBKINETO_DBG_STREAM),
      errnum_(errnum),
      messageSeverity_(severity) {
  buf_ << toString((LoggerOutputType)severity) << ":";

  const auto tt =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  const char* file = strrchr(filePath, '/');
  buf_ << fmt::format("{:%Y-%m-%d %H:%M:%S}", fmt::localtime(tt)) << " "
       << processId() << ":" << systemThreadId() << " "
       << (file ? file + 1 : filePath) << ":" << line << "] ";
}

Logger::~Logger() {
#ifdef __linux__
  if (errnum_ != 0) {
    thread_local char buf[1024];
    buf_ << " : " << strerror_r(errnum_, buf, sizeof(buf));
  }
#endif

  {
    std::lock_guard<std::mutex> guard(loggerObserversMutex());
    for (auto* observer : loggerObservers()) {
      // Output to observers. Current Severity helps keep track of which bucket
      // the output goes.
      if (observer) {
        observer->write(buf_.str(), (LoggerOutputType)messageSeverity_);
      }
    }
  }

  // Finally, print to terminal or console.
  out_ << buf_.str() << std::endl;
}

void Logger::setVerboseLogModules(const std::vector<std::string>& modules) {
  uint64_t mask = 0;
  if (modules.empty()) {
    mask = ~0ull;
  } else {
    for (const std::string& name : modules) {
      mask |= hash(name.c_str());
    }
  }
  verboseLogModules_ = mask;
}

void Logger::addLoggerObserver(ILoggerObserver* observer) {
  if (observer == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  loggerObservers().insert(observer);
}

void Logger::removeLoggerObserver(ILoggerObserver* observer) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  loggerObservers().erase(observer);
}

void Logger::addLoggerObserverDevice(int64_t device) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->addDevice(device);
  }
}

void Logger::addLoggerObserverEventCount(int64_t count) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->addEventCount(count);
  }
}

void Logger::setLoggerObserverTraceDurationMS(int64_t duration) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->setTraceDurationMS(duration);
  }
}

void Logger::setLoggerObserverTraceID(const std::string& tid) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->setTraceID(tid);
  }
}

void Logger::setLoggerObserverGroupTraceID(const std::string& gtid) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->setGroupTraceID(gtid);
  }
}

void Logger::addLoggerObserverDestination(const std::string& dest) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->addDestination(dest);
  }
}

void Logger::setLoggerObserverOnDemand() {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->setTriggerOnDemand();
  }
}

void Logger::addLoggerObserverAddMetadata(
    const std::string& key,
    const std::string& value) {
  std::lock_guard<std::mutex> guard(loggerObserversMutex());
  for (auto observer : loggerObservers()) {
    observer->addMetadata(key, value);
  }
}

} // namespace KINETO_NAMESPACE
