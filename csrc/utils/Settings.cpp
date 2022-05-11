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
 * IPEX_XPU_BACKEND:
 *    Default = 0 (XB_GPU), Set XPU_BACKEND as global IPEX backend
 * IPEX_XPU_SYNC_MODE:
 *    Default = 0, Set 1 to enforce synchronization execution mode
 * IPEX_TILE_AS_DEVICE:
 *    Default = 1, Set 0 to disable tile partition and map per root device
 * IPEX_LAYOUT_OPT:
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
#define DPCPP_ENV_TYPE_DEF(type, name, val, show) \
  auto type = [&]() -> std::optional<int> {       \
    auto env = std::getenv("IPEX_" #name);        \
    std::optional<int> _##type;                   \
    try {                                         \
      _##type = std::stoi(env, 0, 10);            \
    } catch (...) {                               \
      _##type = std::nullopt;                     \
    }                                             \
    if (show) {                                   \
      std::cerr << " ** IPEX_" << #name << ": ";  \
      if (_##type.has_value()) {                  \
        std::cerr << _##type.value();             \
      } else {                                    \
        std::cerr << val;                         \
      }                                           \
      std::cerr << std::endl;                     \
    }                                             \
    return _##type;                               \
  }()

  DPCPP_ENV_TYPE_DEF(show_option, SHOW_OPTION, 0, false);
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
  DPCPP_ENV_TYPE_DEF(env_verbose_level, VERBOSE, verbose_level, show_opt);
  if (env_verbose_level.has_value()) {
    verbose_level = env_verbose_level.value();
  }

  /* Not ready so far.
  xpu_backend = XPU_BACKEND::XB_GPU;
  DPCPP_ENV_TYPE_DEF(env_xpu_backend, XPU_BACKEND, xpu_backend, show_opt);
  if (env_xpu_backend.has_value()
      && ((env_xpu_backend.value() >= XPU_BACKEND::XB_GPU)
        && (env_xpu_backend.value() < XPU_BACKEND::XB_MAX))) {
    xpu_backend = static_cast<XPU_BACKEND>(env_xpu_backend.value());
  }
  */

  xpu_sync_mode_enabled = false;
  DPCPP_ENV_TYPE_DEF(
      env_xpu_sync_mode, XPU_SYNC_MODE, xpu_sync_mode_enabled, show_opt);
  if (env_xpu_sync_mode.has_value() && (env_xpu_sync_mode.value() != 0)) {
    xpu_sync_mode_enabled = true;
  }

  tile_as_device_enabled = true;
  DPCPP_ENV_TYPE_DEF(
      env_tile_as_device, TILE_AS_DEVICE, tile_as_device_enabled, show_opt);
  if (env_tile_as_device.has_value() && (env_tile_as_device.value() == 0)) {
    tile_as_device_enabled = false;
  }

  layout_opt_enabled = false;
  DPCPP_ENV_TYPE_DEF(env_layout_opt, LAYOUT_OPT, layout_opt_enabled, show_opt);
  if (env_layout_opt.has_value() && (env_layout_opt.value() != 0)) {
    layout_opt_enabled = true;
  }

  /* Not ready so far.
  tf32_mode_enabled = false;
  DPCPP_ENV_TYPE_DEF(env_tf32_mode, TF32_MODE, tf32_mode_enabled, show_opt);
  if (env_tf32_mode.has_value() && (env_tf32_mode.value() != 0)) {
    tf32_mode_enabled = true;
  }
  */

#ifdef BUILD_SIMPLE_TRACE
  simple_trace_enabled = false;
  DPCPP_ENV_TYPE_DEF(
      env_simple_trace, SIMPLE_TRACE, simple_trace_enabled, show_opt);
  if (env_simple_trace.has_value() && (env_simple_trace.value() != 0)) {
    simple_trace_enabled = true;
  }
#endif

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

XPU_BACKEND Settings::get_xpu_backend() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return xpu_backend;
}

void Settings::set_xpu_backend(XPU_BACKEND backend) {
  std::lock_guard<std::mutex> lock(s_mutex);
  xpu_backend = backend;
}

bool Settings::is_xpu_sync_mode_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return xpu_sync_mode_enabled;
}

void Settings::enable_xpu_sync_mode() {
  std::lock_guard<std::mutex> lock(s_mutex);
  xpu_sync_mode_enabled = true;
}

void Settings::disable_xpu_sync_mode() {
  std::lock_guard<std::mutex> lock(s_mutex);
  xpu_sync_mode_enabled = false;
}

bool Settings::is_tile_as_device_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return tile_as_device_enabled;
}

bool Settings::is_layout_opt_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return layout_opt_enabled;
}

void Settings::enable_layout_opt() {
  std::lock_guard<std::mutex> lock(s_mutex);
  layout_opt_enabled = true;
}

void Settings::disable_layout_opt() {
  std::lock_guard<std::mutex> lock(s_mutex);
  layout_opt_enabled = false;
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

#ifdef BUILD_SIMPLE_TRACE
bool Settings::is_simple_trace_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return simple_trace_enabled;
}

void Settings::enable_simple_trace() {
  std::lock_guard<std::mutex> lock(s_mutex);
  simple_trace_enabled = true;
}

void Settings::disable_simple_trace() {
  std::lock_guard<std::mutex> lock(s_mutex);
  simple_trace_enabled = false;
}
#endif

} // namespace dpcpp
} // namespace xpu
