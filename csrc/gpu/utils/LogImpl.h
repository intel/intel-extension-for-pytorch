#pragma once

#include <spdlog/spdlog.h>
#include <utils/Settings.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include "spdlog/cfg/env.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"

using namespace std::chrono;

spdlog::level::level_enum get_log_level_from_int(int level);

void log_result(int log_level, std::string message);

// This class is a thread safe singleton class, used for log related settings
struct IPEXLoggingSetting {
  bool enable_logging;
  spdlog::level::level_enum logging_level;
  bool enable_console;
  bool enable_file;

  // output file path for ipex log, for relative file path, it is based on
  // current path location
  std::string output_file_path;

  // for split file size or rotate file size, it is recorded by mb, if it is -1,
  // means it is unuse, default will use rotate_file_size for 10mb if the
  // output_file_path is set
  int rotate_file_size;
  int split_file_size;

  std::vector<std::string> log_component;
  std::vector<std::string> log_sub_component;

  // delete all copy and move functions
  IPEXLoggingSetting(const IPEXLoggingSetting&) = delete;
  IPEXLoggingSetting& operator=(const IPEXLoggingSetting&) = delete;
  IPEXLoggingSetting(IPEXLoggingSetting&&) = delete;
  IPEXLoggingSetting& operator=(IPEXLoggingSetting&&) = delete;

  static IPEXLoggingSetting& get_instance() {
    thread_local static IPEXLoggingSetting instance;
    return instance;
  }

  IPEXLoggingSetting();
};

inline bool should_output(
    std::string message_component,
    std::string message_sub_component) {
  std::vector<std::string> setting_component =
      IPEXLoggingSetting::get_instance().log_component;
  std::vector<std::string> setting_sub_component =
      IPEXLoggingSetting::get_instance().log_sub_component;

  if (std::find(setting_component.begin(), setting_component.end(), "ALL") !=
      setting_component.end()) {
    return true;
  }

  if (std::find(
          setting_component.begin(),
          setting_component.end(),
          message_component) != setting_component.end()) {
    // only SYNGRAPH will use sub component
    if (message_component != "SYNGRAPH") {
      return true;
    } else {
      return !(
          std::find(
              setting_sub_component.begin(),
              setting_sub_component.end(),
              message_sub_component) == setting_sub_component.end());
    }
  }
  return false;
}

inline const std::set<std::string> COMPONENT_SET =
    {"ALL", "OPS", "SYNGRAPH", "MEMORY", "RUNTIME"};

template <typename... Args>
inline std::string& pre_format(
    std::string& s,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  auto out = std::vector<char>();
  fmt::format_to(
      std::back_inserter(out), fmt_message, std::forward<Args>(args)...);

  s += std::string{out.begin(), out.end()};

  return s;
}

class EventMessage {
 public:
  EventMessage();
  EventMessage(
      std::string& _event_id,
      std::string& _step_id,
      std::string& _message,
      uint64_t _timestamp)
      : event_id(_event_id),
        step_id(_step_id),
        message(_message),
        timestamp(_timestamp) {}

  std::string event_id;
  std::string step_id;
  std::string message;
  uint64_t timestamp;
};

class EventLogger {
 public:
  EventLogger() {
    this->message_queue = {};
  };
  ~EventLogger(){};

  template <typename... Args>
  void add_event(
      std::string log_component,
      std::string log_sub_component,
      std::string event_id,
      std::string step_id,
      spdlog::format_string_t<Args...> fmt_message,
      Args&&... args) {
    {
      std::string log_message = "";
      pre_format(log_message, fmt_message, std::forward<Args>(args)...);

      uint64_t nanoseconds =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();

      auto should_log = should_output(log_component, log_sub_component);

      if (should_log) {
        this->message_queue.push_back(
            EventMessage(event_id, step_id, log_message, nanoseconds));
      }
    }
  }
  void print_result(int log_level);

  void print_verbose(
      int log_level,
      const std::string& kernel_name,
      const uint64_t event_duration);

  void print_verbose_ext(
      int log_level,
      const std::string& kernel_name,
      const uint64_t event_duration);

  std::deque<EventMessage> message_queue;
};

class BasicLogger {
 public:
  BasicLogger() {
    if (!IPEXLoggingSetting::get_instance().enable_logging) {
      return;
    }

    std::vector<spdlog::sink_ptr> sinks;
    if (IPEXLoggingSetting::get_instance().enable_console) {
      auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
      console_sink->set_level(IPEXLoggingSetting::get_instance().logging_level);
      console_sink->set_pattern("[%c %z] [%l] [thread %t] %v");
      sinks.push_back(console_sink);
    }
    if (IPEXLoggingSetting::get_instance().enable_file) {
      // for rotate file
      auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
          IPEXLoggingSetting::get_instance().output_file_path,
          IPEXLoggingSetting::get_instance().rotate_file_size,
          1);
      // for split file, will split maximum 10 times
      if (IPEXLoggingSetting::get_instance().split_file_size > 0) {
        file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            IPEXLoggingSetting::get_instance().output_file_path,
            IPEXLoggingSetting::get_instance().split_file_size,
            10);
      }
      file_sink->set_level(IPEXLoggingSetting::get_instance().logging_level);
      file_sink->set_pattern("[%c %z] [%l] [thread %t] %v");
      sinks.push_back(file_sink);
    };

    auto logger = std::make_shared<spdlog::logger>(
        "BasicLogger", begin(sinks), end(sinks));
    logger->set_level(IPEXLoggingSetting::get_instance().logging_level);
    logger->set_pattern("[%c %z] [%l] [thread %t] %v");
    spdlog::set_default_logger(logger);
  }

  static BasicLogger& get_instance() {
    static BasicLogger basic_logger;
    return basic_logger;
  }

  static void update_logger();

 private:
  ~BasicLogger() = default;
  BasicLogger(const BasicLogger&) = delete;
  BasicLogger& operator=(const BasicLogger&) = delete;
};

inline static std::map<std::string, EventLogger> event_logger_map{};

inline EventLogger get_event_logger(std::string event_id) {
  for (auto it = event_logger_map.begin(); it != event_logger_map.end(); ++it) {
    auto value = it->second.message_queue;
  }

  auto find_result = std::find_if(
      event_logger_map.begin(),
      event_logger_map.end(),
      [event_id](const auto& mo) { return mo.first == event_id; });

  event_logger_map.find(event_id);
  if (find_result != event_logger_map.end()) {
    return event_logger_map[event_id];
  } else {
    EventLogger logger;
    event_logger_map.insert(make_pair(event_id, logger));
    return logger;
  }
};

inline void put_event_logger(EventLogger event_logger, std::string event_id) {
  event_logger_map[event_id] = event_logger;
}

template <typename... Args>
inline void log_result_with_args(
    int log_level,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  auto out = std::vector<char>();
  fmt::format_to(
      std::back_inserter(out), fmt_message, std::forward<Args>(args)...);

  std::string s = std::string{out.begin(), out.end()};

  switch (log_level) {
    case 0:
      spdlog::trace(s);
      break;
    case 1:
      spdlog::debug(s);
      break;
    case 2:
      spdlog::info(s);
      break;
    case 3:
      spdlog::warn(s);
      break;
    case 4:
      spdlog::error(s);
      break;
    case 5:
      spdlog::critical(s);
      break;
    default:
      throw std::runtime_error("USING error log level for spdlog");
  }
}

template <typename... Args>
inline void log_info(
    int log_level,
    std::string log_component,
    std::string log_subComponent,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  if (COMPONENT_SET.find(log_component) == COMPONENT_SET.end()) {
    throw std::runtime_error(
        "Invalid logging component, for logging component must be ALL, KERNEL, SYNGRAPH, MEMORY, RUNTIME");
  }

  if (IPEXLoggingSetting::get_instance().enable_logging) {
    auto should_log = should_output(log_component, log_subComponent);
    if (should_log) {
      log_result_with_args(log_level, fmt_message, std::forward<Args>(args)...);
    }
  }
}
