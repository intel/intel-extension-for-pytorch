#include "LogImpl.h"

spdlog::level::level_enum get_log_level_from_int(int level) {
  if (level == 0) {
    return spdlog::level::trace;
  } else if (level == 1) {
    return spdlog::level::debug;
  } else if (level == 2) {
    return spdlog::level::info;
  } else if (level == 3) {
    return spdlog::level::warn;
  } else if (level == 4) {
    return spdlog::level::err;
  } else if (level == 5) {
    return spdlog::level::critical;
  } else {
    throw std::runtime_error(
        "USING error log level for IPEX_LOGGING, log level should be -1 to 5, but met " +
        std::string{level});
  }
}

// log level mapping between number and level are as follow:
// 0 -> trace
// 1 -> debug
// 2 -> info
// 3 -> warn
// 4 -> error
// 5 -> critical
void log_result(int log_level, std::string message) {
  switch (log_level) {
    case 0:
      spdlog::trace(message);
      break;
    case 1:
      spdlog::debug(message);
      break;
    case 2:
      spdlog::info(message);
      break;
    case 3:
      spdlog::warn(message);
      break;
    case 4:
      spdlog::error(message);
      break;
    case 5:
      spdlog::critical(message);
      break;
    default:
      throw std::runtime_error("USING error log level for spdlog");
  }
}

IPEXLoggingSetting::IPEXLoggingSetting() {
  int log_level = xpu::dpcpp::Settings::I().get_log_level();

  if (log_level == -1) {
    this->enable_logging = false;
    return;
  } else {
    this->enable_logging = true;
    this->logging_level = get_log_level_from_int(log_level);
  }

  this->output_file_path = xpu::dpcpp::Settings::I().get_log_output_file_path();

  if (this->output_file_path.size() > 0) {
    this->enable_file = true;
    this->enable_console = false;
  } else {
    this->enable_file = false;
    this->enable_console = true;
  }

  if (this->enable_file) {
    int rotate_file_size = xpu::dpcpp::Settings::I().get_log_rotate_file_size();
    int split_file_size = xpu::dpcpp::Settings::I().get_log_split_file_size();
    if (rotate_file_size == -1 && split_file_size == -1) {
      this->rotate_file_size = 10 * 1024 * 1024;
    } else if (rotate_file_size != -1) {
      this->rotate_file_size = rotate_file_size * 1024 * 1024;
    } else if (split_file_size != -1) {
      this->split_file_size = split_file_size * 1024 * 1024;
    } else {
      throw std::runtime_error("Cannot use both rotate file and split file");
    }
  }

  std::string log_component_str = xpu::dpcpp::Settings::I().get_log_component();

  // For the smallest validate log_component is "ALL" or "OPS", should be larger
  // than 3
  if (log_component_str.size() > 3) {
    auto index = log_component_str.find("/");
    if (index != std::string::npos) {
      std::string log_component_sub_str = log_component_str.substr(0, index);
      std::string log_sub_component_sub_str =
          log_component_str.substr(index + 1, log_component_str.size());

      this->log_component = std::vector<std::string>();
      this->log_sub_component = std::vector<std::string>();

      int log_component_sub_num = std::count(
          log_component_sub_str.begin(), log_component_sub_str.end(), ';');
      int log_sub_component_sub_num = std::count(
          log_sub_component_sub_str.begin(),
          log_sub_component_sub_str.end(),
          ';');

      for (int i = 0; i < log_component_sub_num; i++) {
        index = log_component_sub_str.find(";");
        this->log_component.push_back(log_component_sub_str.substr(0, index));
        log_component_sub_str = log_component_sub_str.substr(
            index + 1, log_component_sub_str.size());
      }
      this->log_component.push_back(log_component_sub_str);

      for (int i = 0; i < log_sub_component_sub_num; i++) {
        index = log_sub_component_sub_str.find(";");
        this->log_sub_component.push_back(
            log_sub_component_sub_str.substr(0, index));
        log_sub_component_sub_str = log_sub_component_sub_str.substr(
            index + 1, log_sub_component_sub_str.size());
      }
      this->log_sub_component.push_back(log_sub_component_sub_str);

    } else {
      this->log_component = std::vector<std::string>();

      int log_component_sub_num =
          std::count(log_component_str.begin(), log_component_str.end(), ';');
      for (int i = 0; i < log_component_sub_num; i++) {
        index = log_component_str.find(";");
        this->log_component.push_back(log_component_str.substr(0, index));
        log_component_str =
            log_component_str.substr(index + 1, log_component_str.size());
      }
      this->log_component.push_back(log_component_str);
    }
  } else {
    // set default log_component = ALL, and without sub_component
    this->log_component = this->log_component.size() == 0
        ? std::vector<std::string>{std::string("ALL")}
        : this->log_component;
    this->log_sub_component = this->log_sub_component.size() == 0
        ? std::vector<std::string>{std::string("")}
        : this->log_sub_component;
    ;
  }

  return;
}

void EventLogger::print_verbose_ext(
    int log_level,
    const std::string& kernel_name,
    const uint64_t event_duration) {
  auto start_msg = this->message_queue[0];
  auto start_barrier_msg = this->message_queue[1];
  auto submit_msg = this->message_queue[2];
  auto end_barrier_msg = this->message_queue[3];
  auto event_wait_msg = this->message_queue[4];

  std::stringstream ss;
  ss << kernel_name << ": submit = "
     << static_cast<float>(
            (submit_msg.timestamp - start_msg.timestamp) / 1000.0);
  ss << " us , event wait = "
     << static_cast<float>(
            (event_wait_msg.timestamp - submit_msg.timestamp) / 1000.0);
  ss << " us , event duration = " << event_duration;
  ss << " us , total = "
     << static_cast<float>(
            (event_wait_msg.timestamp - start_msg.timestamp) / 1000);
  ss << " us , barrier wait time = "
     << static_cast<float>(
            (end_barrier_msg.timestamp - start_barrier_msg.timestamp) / 1000);
  ss << " us";
  auto log_str = ss.str();
  log_result_with_args(log_level, log_str);
}

// this is for log message format as previous IPEX_VERBOSE
void EventLogger::print_verbose(
    int log_level,
    const std::string& kernel_name,
    const uint64_t event_duration) {
  auto start_msg = this->message_queue[0];
  auto submit_msg = this->message_queue[1];
  auto event_wait_msg = this->message_queue[2];

  std::stringstream ss;
  ss << kernel_name << ": submit = "
     << static_cast<float>(
            (submit_msg.timestamp - start_msg.timestamp) / 1000.0);
  ss << " us , event wait = "
     << static_cast<float>(
            (event_wait_msg.timestamp - submit_msg.timestamp) / 1000.0);
  ss << " us , event duration = " << event_duration;
  ss << " us , total = "
     << static_cast<float>(
            (event_wait_msg.timestamp - start_msg.timestamp) / 1000);
  ss << " us";
  auto log_str = ss.str();
  log_result_with_args(log_level, log_str);
}

void EventLogger::print_result(int log_level) {
  assert(this->message_queue.size() >= 2);
  auto first = this->message_queue.front();
  auto last = this->message_queue.back();
  auto cost_time = last.timestamp - first.timestamp;
  auto event_id = this->message_queue.front().event_id;
  std::ostringstream stringStream;
  stringStream << "Total event cost time is:"
               << static_cast<float>(cost_time / 1000.0) << " us.";
  log_result(log_level, stringStream.str());
  while (!this->message_queue.empty()) {
    auto msg = this->message_queue.front().message;

    log_result(log_level, this->message_queue.front().message);
    auto this_time = this->message_queue.front().timestamp;
    auto this_step = this->message_queue.front().step_id;
    message_queue.pop_front();
    // have more than 2 step, need to calculate timestamp between each step.
    if (this->message_queue.size() >= 2) {
      auto next_time = this->message_queue.front().timestamp;
      auto next_step = this->message_queue.front().step_id;
      // inside IPEX_LOGGING we are using nanoseconds, 1ns = 0.001us, cast to us
      // here
      auto time_step = static_cast<float>((next_time - this_time) / 1000);
      log_result_with_args(
          log_level,
          "{} between step:{} and step:{} usage time {} us",
          event_id,
          this_step,
          next_step,
          time_step);
      this_time = next_time;
      this_step = next_step;
    }
  }
}

void BasicLogger::update_logger() {
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
  }

  auto logger =
      std::make_shared<spdlog::logger>("BasicLogger", begin(sinks), end(sinks));
  logger->set_level(IPEXLoggingSetting::get_instance().logging_level);
  logger->set_pattern("[%c %z] [%l] [thread %t] %v");
  spdlog::set_default_logger(logger);
}
