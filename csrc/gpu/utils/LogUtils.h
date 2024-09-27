#pragma once

#include "LogImpl.h"

/*
 * The following functions are for simple log functions for ipex logging.
 * Currently, ipex logging support six level for logging, from trace to
 * critical. The basic usage for each parameter are as below:
 *
 * log_component: we have four basic log_component, if you would like to
 * log all components please use "ALL", if you would like to log several
 * log_component, pls use ';' as sepreator, such as "OPS;RUNTIME"
 *
 * log_sub_component:  This only works when log_component is SYNGRAPH,
 * and it is not fixed type restrictions
 *
 * fmt_message: message format template, implemented by fmt library,
 * quiet like python format, you  can use {} as a placeholder
 *
 * args: args for fmt_message
 */

template <typename... Args>
inline void IPEX_TRACE_LOG(
    std::string log_component,
    std::string log_sub_component,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  log_info(
      0,
      log_component,
      log_sub_component,
      fmt_message,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_DEBUG_LOG(
    std::string log_component,
    std::string log_sub_component,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  log_info(
      1,
      log_component,
      log_sub_component,
      fmt_message,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_INFO_LOG(
    std::string log_component,
    std::string log_sub_component,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  log_info(
      2,
      log_component,
      log_sub_component,
      fmt_message,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_WARN_LOG(
    std::string log_component,
    std::string log_sub_component,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  log_info(
      3,
      log_component,
      log_sub_component,
      fmt_message,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_ERR_LOG(
    std::string log_component,
    std::string log_sub_component,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  log_info(
      4,
      log_component,
      log_sub_component,
      fmt_message,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_FATAL_LOG(
    std::string log_component,
    std::string log_sub_component,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  log_info(
      5,
      log_component,
      log_sub_component,
      fmt_message,
      std::forward<Args>(args)...);
}

/*
 * The following functions are for event log functions for ipex logging.
 * Currently, ipex logging support six level for logging, from trace to
 * critical. The basic usage for each parameter are as below:
 *
 * log_component: we have four basic log_component,
 * if you would like to log all components, please use "ALL"
 *
 * log_sub_component: This only works when log_component is SYNGRAPH,
 * and it is not fixed type restrictions
 *
 * event_id:  An id for the whole event,
 * should be the same when logging the same event
 *
 * step_id: A step id for single step.
 *
 * fmt_message: message format template, implemented by fmt library ,
 * quiet like python format, you can use {} as a placeholder
 *
 * args: args for fmt_message
 *
 * At the end of each event log, IPEX_XXX_EVENT_END function is used for the end
 * of an event log, the event_id should keep align with previous event_id,
 * XXX is used for tag the whole event log level,
 */

template <typename... Args>
inline void IPEX_EVENT_LOG(
    std::string log_component,
    std::string log_sub_component,
    std::string event_id,
    std::string step_id,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  auto logger = get_event_logger(event_id);

  logger.add_event(
      log_component,
      log_sub_component,
      event_id,
      step_id,
      fmt_message,
      std::forward<Args>(args)...);

  put_event_logger(logger, event_id);
};

template <typename... Args>
inline void IPEX_EVENT_END(
    std::string log_component,
    std::string log_sub_component,
    std::string event_id,
    std::string step_id,
    spdlog::format_string_t<Args...> fmt_message,
    int log_level,
    Args&&... args) {
  auto logger = get_event_logger(event_id);
  logger.add_event(
      log_component,
      log_sub_component,
      event_id,
      step_id,
      fmt_message,
      std::forward<Args>(args)...);
  put_event_logger(logger, event_id);
  logger.print_result(log_level);
};

template <typename... Args>
inline void IPEX_TRACE_EVENT_END(
    std::string log_component,
    std::string log_sub_component,
    std::string event_id,
    std::string step_id,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  IPEX_EVENT_END(
      log_component,
      log_sub_component,
      event_id,
      step_id,
      fmt_message,
      0,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_DEBUG_EVENT_END(
    std::string log_component,
    std::string log_sub_component,
    std::string event_id,
    std::string step_id,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  IPEX_EVENT_END(
      log_component,
      log_sub_component,
      event_id,
      step_id,
      fmt_message,
      1,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_INFO_EVENT_END(
    std::string log_component,
    std::string log_sub_component,
    std::string event_id,
    std::string step_id,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  IPEX_EVENT_END(
      log_component,
      log_sub_component,
      event_id,
      step_id,
      fmt_message,
      2,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_WARN_EVENT_END(
    std::string log_component,
    std::string log_sub_component,
    std::string event_id,
    std::string step_id,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  IPEX_EVENT_END(
      log_component,
      log_sub_component,
      event_id,
      step_id,
      fmt_message,
      3,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_ERROR_EVENT_END(
    std::string log_component,
    std::string log_sub_component,
    std::string event_id,
    std::string step_id,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  IPEX_EVENT_END(
      log_component,
      log_sub_component,
      event_id,
      step_id,
      fmt_message,
      4,
      std::forward<Args>(args)...);
}

template <typename... Args>
inline void IPEX_INFO_CRITAL_END(
    std::string log_component,
    std::string log_sub_component,
    std::string event_id,
    std::string step_id,
    spdlog::format_string_t<Args...> fmt_message,
    Args&&... args) {
  IPEX_EVENT_END(
      log_component,
      log_sub_component,
      event_id,
      step_id,
      fmt_message,
      5,
      std::forward<Args>(args)...);
}
