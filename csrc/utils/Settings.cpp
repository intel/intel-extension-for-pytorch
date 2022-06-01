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
 * IPEX_FP32_MATH_MODE:
 *    Default = 0, Set values for FP32 math mode (0: FP32, 1: TF32, 2: BF32)
 *
 * XPU private optionos:
 *   IPEX_XPU_BACKEND:
 *      Default = 0 (XB_GPU), Set XPU_BACKEND as global IPEX backend
 *   IPEX_XPU_SYNC_MODE:
 *      Default = 0, Set 1 to enforce synchronization execution mode
 *   IPEX_TILE_AS_DEVICE:
 *      Default = 1, Set 0 to disable tile partition and map per root device
 *   IPEX_SIMPLE_TRACE:
 *      Default = 0, Set 1 to enable simple trace for all operators*
 *
 * Experimental options:
 *   IPEX_XPU_ONEDNN_LAYOUT:
 *      Default = 0, Set 1 to enable onednn specific layouts
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

  xpu_backend = XPU_BACKEND::XB_GPU;
  DPCPP_ENV_TYPE_DEF(env_xpu_backend, XPU_BACKEND, xpu_backend, show_opt);
  if (env_xpu_backend.has_value() &&
      ((env_xpu_backend.value() >= XPU_BACKEND::XB_GPU) &&
       (env_xpu_backend.value() < XPU_BACKEND::XB_MAX))) {
    xpu_backend = static_cast<XPU_BACKEND>(env_xpu_backend.value());
  }

  sync_mode_enabled = false;
  DPCPP_ENV_TYPE_DEF(env_sync_mode, XPU_SYNC_MODE, sync_mode_enabled, show_opt);
  if (env_sync_mode.has_value() && (env_sync_mode.value() != 0)) {
    sync_mode_enabled = true;
  }

  tile_as_device_enabled = true;
  DPCPP_ENV_TYPE_DEF(
      env_tile_as_device, TILE_AS_DEVICE, tile_as_device_enabled, show_opt);
  if (env_tile_as_device.has_value() && (env_tile_as_device.value() == 0)) {
    tile_as_device_enabled = false;
  }

  onednn_layout_enabled = false;
  DPCPP_ENV_TYPE_DEF(
      env_onednn_layout, XPU_ONEDNN_LAYOUT, onednn_layout_enabled, show_opt);
  if (env_onednn_layout.has_value() && (env_onednn_layout.value() != 0)) {
    onednn_layout_enabled = true;
  }

  fp32_math_mode = FP32_MATH_MODE::FMM_FP32;
  DPCPP_ENV_TYPE_DEF(env_math_mode, FP32_MATH_MODE, fp32_math_mode, show_opt);
  if (env_math_mode.has_value() &&
      ((env_math_mode.value() >= FP32_MATH_MODE::FMM_FP32) &&
       (env_math_mode.value() < FP32_MATH_MODE::FMM_MAX))) {
    fp32_math_mode = static_cast<FP32_MATH_MODE>(env_math_mode.value());
  }

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

bool Settings::set_verbose_level(int level) {
  std::lock_guard<std::mutex> lock(s_mutex);
  verbose_level = level;
  return true;
}

XPU_BACKEND Settings::get_backend() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return xpu_backend;
}

bool Settings::set_backend(XPU_BACKEND backend) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if ((backend >= XPU_BACKEND::XB_GPU) && (backend < XPU_BACKEND::XB_MAX)) {
    xpu_backend = backend;
    return true;
  }
  return false;
}

bool Settings::is_sync_mode_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return sync_mode_enabled;
}

void Settings::enable_sync_mode() {
  std::lock_guard<std::mutex> lock(s_mutex);
  sync_mode_enabled = true;
}

void Settings::disable_sync_mode() {
  std::lock_guard<std::mutex> lock(s_mutex);
  sync_mode_enabled = false;
}

bool Settings::is_tile_as_device_enabled() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return tile_as_device_enabled;
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

FP32_MATH_MODE Settings::get_fp32_math_mode() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return fp32_math_mode;
}

bool Settings::set_fp32_math_mode(FP32_MATH_MODE math_mode) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if ((math_mode >= FP32_MATH_MODE::FMM_FP32) &&
      (math_mode < FP32_MATH_MODE::FMM_MAX)) {
    fp32_math_mode = math_mode;
    return true;
  }
  return false;
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
