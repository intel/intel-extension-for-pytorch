#include <oneDNN/Runtime.h>
#include <utils/Settings.h>
#include <utils/oneMKLUtils.h>

#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>

namespace xpu {
namespace dpcpp {

/*
 * All available launch options for IPEX
 *
 * IPEX_SHOW_OPTION:
 *    Default = 0, Set 1 to show all launch option values
 * IPEX_VERBOSE:
 *    Default = 0, Set verbose level with synchronization execution mode
 * IPEX_WARNING:
 *    Default = 0, Set warning level as int value for IPEX log lines
 * IPEX_XPU_BACKEND:
 *    Default = 0 (XB_GPU), Set XPU_BACKEND as global IPEX backend
 * IPEX_FORCE_SYNC:
 *    Default = 0, Set 1 to enforce synchronization execution mode
 * IPEX_DISABLE_PROFILING:
 *    Default = 0, Set 1 to disable IPEX event profiling
 * IPEX_DISABLE_TILE_PARTITION:
 *    Default = 0, Set 1 to disable tile partition and map per root device
 * IPEX_ONEDNN_LAYOUT:
 *    Default = 0, Set 1 to enable onednn specific layouts
 * IPEX_TF32_MODE:
 *    Default = 0, Set 1 to enable TF32 mode execution
 */

static std::mutex s_mutex;

static Settings mySettings;

Settings& Settings::I() {
  return mySettings;
}

Settings::Settings() {
#define DPCPP_ENV_TYPE_DEF(type, var, show)     \
  auto type = [&]() -> std::optional<int> {     \
    auto env = std::getenv("IPEX_" #var);       \
    std::optional<int> _##type;                 \
    try {                                       \
      _##type = std::stoi(env, 0, 10);          \
    } catch (...) {                             \
      _##type = std::nullopt;                   \
    }                                           \
    if (show) {                                 \
      std::cerr << " ** IPEX_" << #var << ": "; \
      if (_##type.has_value()) {                \
        std::cerr << _##type.value();           \
      } else {                                  \
        std::cerr << "NIL";                     \
      }                                         \
      std::cerr << std::endl;                   \
    }                                           \
    return _##type;                             \
  }()

  DPCPP_ENV_TYPE_DEF(show_option, SHOW_OPTION, false);
  bool show_opt =
      show_option.has_value() ? (show_option != 0 ? true : false) : false;
  if (show_opt) {
    std::cerr << std::endl
              << " *********************************************************"
              << std::endl
              << " ** The values of all available launch options for IPEX **"
              << std::endl
              << " *********************************************************"
              << std::endl;
  }

  verbose_level = 0;
  DPCPP_ENV_TYPE_DEF(env_verbose_level, VERBOSE, show_opt);
  if (env_verbose_level.has_value()) {
    verbose_level = env_verbose_level.value();
  }

  warning_level = 0;
  DPCPP_ENV_TYPE_DEF(env_warning_level, WARNING, show_opt);
  if (env_warning_level.has_value()) {
    warning_level = env_warning_level.value();
  }

  /* Not ready so far.
  xpu_backend = XPU_BACKEND::XB_GPU;
  DPCPP_ENV_TYPE_DEF(env_xpu_backend, XPU_BACKEND, show_opt);
  if (env_xpu_backend.has_value()
      && ((env_xpu_backend.value() >= XPU_BACKEND::XB_GPU)
        && (env_xpu_backend.value() < XPU_BACKEND::XB_MAX))) {
    xpu_backend = static_cast<XPU_BACKEND>(env_xpu_backend.value());
  }
  */

  force_sync_exec_enabled = false;
  DPCPP_ENV_TYPE_DEF(env_force_sync, FORCE_SYNC, show_opt);
  if (env_force_sync.has_value() && (env_force_sync.value() != 0)) {
    force_sync_exec_enabled = true;
  }

  event_profiling_enabled = true;
  DPCPP_ENV_TYPE_DEF(env_disable_profiling, DISABLE_PROFILING, show_opt);
  if (env_disable_profiling.has_value() &&
      (env_disable_profiling.value() != 0)) {
    event_profiling_enabled = false;
  }

  tile_partition_enabled = true;
  DPCPP_ENV_TYPE_DEF(
      env_disable_tile_partition, DISABLE_TILE_PARTITION, show_opt);
  if (env_disable_tile_partition.has_value() &&
      (env_disable_tile_partition.value() != 0)) {
    tile_partition_enabled = false;
  }

  onednn_layout_enabled = false;
  DPCPP_ENV_TYPE_DEF(env_onednn_layout, ONEDNN_LAYOUT, show_opt);
  if (env_onednn_layout.has_value() && (env_onednn_layout.value() != 0)) {
    onednn_layout_enabled = true;
  }

  /* Not ready so far.
  tf32_mode_enabled = false;
  DPCPP_ENV_TYPE_DEF(env_tf32_mode, TF32_MODE, show_opt);
  if (env_tf32_mode.has_value() && (env_tf32_mode.value() != 0)) {
    tf32_mode_enabled = true;
  }
  */

  if (show_opt) {
    std::cerr << " *********************************************************"
              << std::endl;
  }
}

int Settings::get_verbose_level() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return verbose_level;
}

void Settings::set_verbose_level(int level) {
  std::lock_guard<std::mutex> lock(s_mutex);
  verbose_level = level;
}

int Settings::get_warning_level() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return warning_level;
}

void Settings::set_warning_level(int level) {
  std::lock_guard<std::mutex> lock(s_mutex);
  warning_level = level;
}

XPU_BACKEND Settings::get_xpu_backend() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return xpu_backend;
}

void Settings::set_xpu_backend(XPU_BACKEND backend) {
  std::lock_guard<std::mutex> lock(s_mutex);
  xpu_backend = backend;
}

bool Settings::is_force_sync_exec() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return force_sync_exec_enabled;
}

void Settings::enable_force_sync_exec() {
  std::lock_guard<std::mutex> lock(s_mutex);
  force_sync_exec_enabled = true;
}

void Settings::disable_force_sync_exec() {
  std::lock_guard<std::mutex> lock(s_mutex);
  force_sync_exec_enabled = false;
}

bool Settings::is_event_profiling_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return event_profiling_enabled;
}

void Settings::enable_event_profiling() {
  std::lock_guard<std::mutex> lock(s_mutex);
  event_profiling_enabled = true;
}

void Settings::disable_event_profiling() {
  std::lock_guard<std::mutex> lock(s_mutex);
  event_profiling_enabled = false;
}

bool Settings::is_tile_partition_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return tile_partition_enabled;
}

void Settings::enable_tile_partition() {
  std::lock_guard<std::mutex> lock(s_mutex);
  tile_partition_enabled = true;
}

void Settings::disable_tile_partition() {
  std::lock_guard<std::mutex> lock(s_mutex);
  tile_partition_enabled = false;
}

bool Settings::is_onednn_layout_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return onednn_layout_enabled;
}

void Settings::enable_onednn_layout() {
  std::lock_guard<std::mutex> lock(s_mutex);
  onednn_layout_enabled = true;
}

void Settings::disable_onednn_layout() {
  std::lock_guard<std::mutex> lock(s_mutex);
  onednn_layout_enabled = false;
}

bool Settings::is_tf32_mode_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return tf32_mode_enabled;
}

void Settings::enable_tf32_mode() {
  std::lock_guard<std::mutex> lock(s_mutex);
  tf32_mode_enabled = true;
}

void Settings::disable_tf32_mode() {
  std::lock_guard<std::mutex> lock(s_mutex);
  tf32_mode_enabled = false;
}

bool Settings::set_onednn_verbose(int level) {
  return xpu::oneDNN::set_onednn_verbose(level);
}

bool Settings::set_onemkl_verbose(int level) {
  return xpu::oneMKL::set_onemkl_verbose(level);
}

bool Settings::is_onemkl_enabled() const {
#if defined(USE_ONEMKL)
  return true;
#else
  return false;
#endif
}

bool Settings::is_channels_last_1d_enabled() const {
#if defined(USE_CHANNELS_LAST_1D)
  return true;
#else
  return false;
#endif
}

bool Settings::is_double_disabled() const {
#if defined(BUILD_INTERNAL_DEBUG) && !defined(BUILD_DOUBLE_KERNEL)
  return false;
#else
  return true;
#endif
}

} // namespace dpcpp
} // namespace xpu
