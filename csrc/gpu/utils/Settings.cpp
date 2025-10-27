#include <oneDNN/Runtime.h>
#include <runtime/Device.h>
#include <utils/Settings.h>
#include <utils/oneMKLUtils.h>

#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>

namespace torch_ipex::xpu {
namespace dpcpp {

// clang-format off
/*
 * [Keep the format for automatic doc generation.]
 * All available launch options for IPEX
 * ==========ALL==========
 *   IPEX_FP32_MATH_MODE:
 *      Default = FP32 | Set values for FP32 math mode (valid values: FP32, TF32, BF32). Refer to <a class="reference internal" href="../api_doc.html#_CPPv4N3xpu18set_fp32_math_modeE14FP32_MATH_MODE">API Documentation</a> for details.
 * ==========ALL==========
 *
 * XPU ONLY optionos:
 * ==========GPU==========
 *   IPEX_LOG_LEVEL:
 *      Default = -1 | Set log level to trace the execution and get log information, pls refer to 'ipex_log.md' for different log level.
 *   IPEX_LOG_COMPONENT:
 *      Default = "ALL" | Set IPEX_LOG_COMPONENT = ALL to log all component message.
 *      Use ';' as separator to log more than one components, such as "OPS;RUNTIME".
 *      Use '/' as separator to log subcomponents.
 *   IPEX_LOG_ROTATE_SIZE:
 *      Default = -1 | Set Rotate file size in MB for IPEX_LOG, less than 0 means unuse this setting.
 *   IPEX_LOG_SPLIT_SIZE:
 *      Default = -1 | Set split file size in MB for IPEX_LOG, less than 0 means unuse this setting.
 *   IPEX_LOG_OUTPUT:
 *      Default = "" | Set output file path for IPEX_LOG, default is null
 * ==========GPU==========
 *
 * Internal options:
 * ==========INT==========
 *   IPEX_SHOW_OPTION:
 *      Default = 0 | Set 1 to show all launch option values
 *   IPEX_XPU_ONEDNN_LAYOUT:
 *      Default = 0 | Set 1 to enable onednn specific layouts
 *   IPEX_XPU_BACKEND:
 *      Default = 0 (GPU) | Set XPU_BACKEND as global IPEX backend
 *   IPEX_COMPUTE_ENG:
 *      Default = 0 (RECOMMEND) | Set RECOMMEND to select recommended compute engine
 *      operators: RECOMMEND, BASIC, ONEDNN, ONEMKL, XETLA
 * ==========INT==========
 */
// clang-format on

static std::mutex s_mutex;

static Settings mySettings;

Settings& Settings::I() {
  return mySettings;
}

Settings::Settings() {
#define DPCPP_INIT_ENV_VAL(name, var, etype, show)           \
  do {                                                       \
    auto env = std::getenv(name);                            \
    if (env) {                                               \
      try {                                                  \
        int _ival = std::stoi(env, 0, 10);                   \
        if (_ival <= etype##_MAX && _ival >= etype##_MIN) {  \
          var = static_cast<decltype(var)>(_ival);           \
        }                                                    \
      } catch (...) {                                        \
        try {                                                \
          std::string _sval(env);                            \
          for (int i = etype##_MIN; i <= etype##_MAX; i++) { \
            if (_sval == etype##_STR[i]) {                   \
              var = static_cast<decltype(var)>(i);           \
              break;                                         \
            }                                                \
          }                                                  \
        } catch (...) {                                      \
        }                                                    \
      }                                                      \
    }                                                        \
    if (show) {                                              \
      std::cerr << " ** " << name << ": ";                   \
      if (var <= etype##_MAX && var >= etype##_MIN) {        \
        std::cerr << etype##_STR[var];                       \
      } else {                                               \
        std::cerr << "UNKNOW";                               \
      }                                                      \
      std::cerr << " (= " << var << ")" << std::endl;        \
    }                                                        \
  } while (0)

  ENV_VAL show_opt = ENV_VAL::OFF;
  DPCPP_INIT_ENV_VAL("IPEX_SHOW_OPTION", show_opt, ENV_VAL, false);
  if (show_opt) {
    std::cerr << std::endl
              << " *********************************************************"
              << std::endl
              << " ** The values of all available launch options for IPEX **"
              << std::endl
              << " *********************************************************"
              << std::endl;
  }

  log_level = LOG_LEVEL::DISABLED;
  DPCPP_INIT_ENV_VAL("IPEX_LOG_LEVEL", log_level, LOG_LEVEL, show_opt);

  // get IPEX_LOG_ROTATE_SIZE
  auto IPEX_LOG_ROTATE_SIZE_env = std::getenv("IPEX_LOG_ROTATE_SIZE");
  if (IPEX_LOG_ROTATE_SIZE_env == NULL) {
    log_rotate_size = -1;
  } else {
    log_rotate_size = std::stoi(IPEX_LOG_ROTATE_SIZE_env, 0, 10);
  }

  // get IPEX_LOG_SPLIT_SIZE
  auto IPEX_LOG_SPLIT_SIZE_env = std::getenv("IPEX_LOG_SPLIT_SIZE");
  if (IPEX_LOG_SPLIT_SIZE_env == NULL) {
    log_split_size = -1;
  } else {
    log_split_size = std::stoi(IPEX_LOG_SPLIT_SIZE_env, 0, 10);
  }

  // get IPEX_LOG_OUTPUT
  auto IPEX_LOG_OUTPUT_env = std::getenv("IPEX_LOG_OUTPUT");
  if (IPEX_LOG_OUTPUT_env == NULL) {
    log_output = "";
  } else {
    log_output = std::string(IPEX_LOG_OUTPUT_env);
  }

  // get IPEX_LOG_COMPONENT
  auto IPEX_LOG_COMPONENT_env = std::getenv("IPEX_LOG_COMPONENT");
  if (IPEX_LOG_COMPONENT_env == NULL) {
    log_component = "ALL";
  } else {
    log_component = std::string(IPEX_LOG_COMPONENT_env);
  }

  xpu_backend = XPU_BACKEND::GPU;
  DPCPP_INIT_ENV_VAL("IPEX_XPU_BACKEND", xpu_backend, XPU_BACKEND, show_opt);

  onednn_layout_enabled = ENV_VAL::OFF;
  DPCPP_INIT_ENV_VAL(
      "IPEX_XPU_ONEDNN_LAYOUT", onednn_layout_enabled, ENV_VAL, show_opt);

  compute_eng = COMPUTE_ENG::RECOMMEND;
  DPCPP_INIT_ENV_VAL("IPEX_COMPUTE_ENG", compute_eng, COMPUTE_ENG, show_opt);

  fp32_math_mode = FP32_MATH_MODE::FP32;
  DPCPP_INIT_ENV_VAL(
      "IPEX_FP32_MATH_MODE", fp32_math_mode, FP32_MATH_MODE, show_opt);

  if (show_opt) {
    std::cerr << " *********************************************************"
              << std::endl;
  }
} // namespace dpcpp

bool Settings::has_fp64_dtype(int device_id) {
  return dpcppSupportFP64(device_id);
}

bool Settings::has_2d_block_array(int device_id) {
  return dpcppGetDeviceHas2DBlock(device_id);
}

bool Settings::has_atomic64(int device_id) {
  return dpcppSupportAtomic64(device_id);
}

bool Settings::has_xmx(int device_id) {
  // whether XMX is supported in the specified platform.
  return dpcppGetDeviceHasXMX(device_id);
}

XPU_BACKEND Settings::get_backend() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return xpu_backend;
}

bool Settings::set_backend(XPU_BACKEND backend) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if ((backend >= XPU_BACKEND::GPU) && (backend <= XPU_BACKEND_MAX)) {
    xpu_backend = backend;
    return true;
  }
  return false;
}

COMPUTE_ENG Settings::get_compute_eng() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return compute_eng;
}

bool Settings::set_compute_eng(COMPUTE_ENG eng) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if ((eng >= COMPUTE_ENG::RECOMMEND) && (eng <= COMPUTE_ENG_MAX)) {
    compute_eng = eng;
    return true;
  }
  return false;
}

bool Settings::is_onednn_layout_enabled() const {
  return false;
}

void Settings::enable_onednn_layout() {
  TORCH_WARN_ONCE("oneDNN block format support is deprecated since 2.5");
}

void Settings::disable_onednn_layout() {
  TORCH_WARN_ONCE("oneDNN block format support is deprecated since 2.5");
}

FP32_MATH_MODE Settings::get_fp32_math_mode() const {
  std::lock_guard<std::mutex> lock(s_mutex);
  return fp32_math_mode;
}

bool Settings::set_fp32_math_mode(FP32_MATH_MODE mode) {
  std::lock_guard<std::mutex> lock(s_mutex);
  if ((mode >= FP32_MATH_MODE::FP32) && (mode <= FP32_MATH_MODE_MAX)) {
    fp32_math_mode = mode;
    return true;
  }
  return false;
}

bool Settings::set_onednn_verbose(int level) {
  return torch_ipex::xpu::oneDNN::set_onednn_verbose(level);
}

bool Settings::set_onemkl_verbose(int level) {
  return torch_ipex::xpu::oneMKL::set_onemkl_verbose(level);
}

bool Settings::is_onemkl_enabled() const {
#if defined(USE_ONEMKL)
  return true;
#else
  return false;
#endif
}

bool Settings::is_xetla_enabled() const {
#if defined(USE_XETLA)
  return true;
#else
  return false;
#endif
}

bool Settings::is_ds_kernel_enabled() const {
#ifdef USE_DS_KERNELS
  return true;
#else
  return false;
#endif
}

bool Settings::is_bnb_kernel_enabled() const {
#ifdef USE_BNB_KERNELS
  return true;
#else
  return false;
#endif
}

int64_t Settings::get_compiler_version() const {
  return __INTEL_LLVM_COMPILER;
}

} // namespace dpcpp

/* FIXME: The backend is not ready for now.
 * Do not export to public
XPU_BACKEND get_backend() {
  return dpcpp::Settings::I().get_backend();
}

bool set_backend(XPU_BACKEND backend) {
  return dpcpp::Settings::I().set_backend(backend);
}
*/

FP32_MATH_MODE get_fp32_math_mode() {
  return dpcpp::Settings::I().get_fp32_math_mode();
}

bool set_fp32_math_mode(FP32_MATH_MODE mode) {
  return dpcpp::Settings::I().set_fp32_math_mode(mode);
}

} // namespace torch_ipex::xpu
