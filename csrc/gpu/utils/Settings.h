#pragma once

#include <include/xpu/Settings.h>
#include <utils/Macros.h>

namespace torch_ipex::xpu {

enum ENV_VAL { OFF = 0, ON = 1, ENV_VAL_MIN = OFF, ENV_VAL_MAX = ON };
static const char* ENV_VAL_STR[]{"OFF", "ON"};

enum LOG_LEVEL {
  DISABLED = -1,
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERR = 4,
  FATAL = 5,
  LOG_LEVEL_MAX = FATAL,
  LOG_LEVEL_MIN = DISABLED
};
static const char* LOG_LEVEL_STR[]{
    "DISABLED",
    "TRACE",
    "DEBUG",
    "INFO",
    "WARN",
    "ERR",
    "FATAL"};

enum XPU_BACKEND {
  GPU = 0,
  CPU = 1,
  AUTO = 2,
  XPU_BACKEND_MIN = GPU,
  XPU_BACKEND_MAX = AUTO
};
static const char* XPU_BACKEND_STR[]{"GPU", "CPU", "AUTO"};

enum COMPUTE_ENG {
  RECOMMEND = 0,
  BASIC = 1,
  ONEDNN = 2,
  ONEMKL = 3,
  XETLA = 4,
  COMPUTE_ENG_MIN = RECOMMEND,
  COMPUTE_ENG_MAX = XETLA
};
static const char* COMPUTE_ENG_STR[]{
    "RECOMMEND",
    "BASIC",
    "ONEDNN",
    "ONEMKL",
    "XETLA"};

namespace dpcpp {

class IPEX_API Settings final {
 public:
  Settings();

  bool has_fp64_dtype(int device_id = -1);
  bool has_2d_block_array(int device_id = -1);
  bool has_atomic64(int device_id = -1);
  bool has_xmx(int device_id = -1);

  static Settings& I(); // Singleton

  XPU_BACKEND get_backend() const;
  bool set_backend(XPU_BACKEND backend);

  COMPUTE_ENG get_compute_eng() const;
  bool set_compute_eng(COMPUTE_ENG eng);

  bool is_onednn_layout_enabled() const;
  void enable_onednn_layout();
  void disable_onednn_layout();

  FP32_MATH_MODE get_fp32_math_mode() const;
  bool set_fp32_math_mode(FP32_MATH_MODE mode);

  bool set_onednn_verbose(int level);
  bool set_onemkl_verbose(int level);

  bool is_onemkl_enabled() const;

  bool is_xetla_enabled() const;

  bool is_ds_kernel_enabled() const;
  bool is_bnb_kernel_enabled() const;

  int64_t get_compiler_version() const;

 private:
  LOG_LEVEL log_level;
  std::string log_component;
  std::string log_output;
  int log_rotate_size;
  int log_split_size;
  XPU_BACKEND xpu_backend;
  COMPUTE_ENG compute_eng;
  FP32_MATH_MODE fp32_math_mode;

  ENV_VAL sync_mode_enabled;
  ENV_VAL onednn_layout_enabled;
};

} // namespace dpcpp
} // namespace torch_ipex::xpu
