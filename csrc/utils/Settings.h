#pragma once

#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

enum IPEX_API ENV_VAL { OFF = 0, ON = 1, ENV_VAL_MAX = ON };
static const char* IPEX_API ENV_VAL_STR[]{"OFF", "ON"};

enum IPEX_API VERBOSE_LEVEL {
  DISABLE = 0,
  DEBUG = 1,
  VERBOSE_LEVEL_MAX = DEBUG
};
static const char* IPEX_API VERBOSE_LEVEL_STR[]{"DISABLE", "DEBUG"};

enum IPEX_API XPU_BACKEND {
  GPU = 0,
  CPU = 1,
  AUTO = 2,
  XPU_BACKEND_MAX = AUTO
};
static const char* IPEX_API XPU_BACKEND_STR[]{"GPU", "CPU", "AUTO"};

enum IPEX_API FP32_MATH_MODE {
  FP32 = 0,
  TF32 = 1,
  BF32 = 2,
  FP32_MATH_MODE_MAX = BF32
};
static const char* IPEX_API IPEX_API
    FP32_MATH_MODE_STR[]{"FP32", "TF32", "BF32"};

class IPEX_API Settings final {
 public:
  Settings();

  static Settings& I(); // Singleton

  int get_verbose_level() const;
  bool set_verbose_level(int level);

  XPU_BACKEND get_backend() const;
  bool set_backend(XPU_BACKEND backend);

  bool is_sync_mode_enabled() const;
  void enable_sync_mode();
  void disable_sync_mode();

  bool is_tile_as_device_enabled() const;

  bool is_onednn_layout_enabled() const;
  void enable_onednn_layout();
  void disable_onednn_layout();

  FP32_MATH_MODE get_fp32_math_mode() const;
  bool set_fp32_math_mode(FP32_MATH_MODE mode);

  bool set_onednn_verbose(int level);
  bool set_onemkl_verbose(int level);

  bool is_onemkl_enabled() const;

  bool is_channels_last_1d_enabled() const;

#ifdef BUILD_SIMPLE_TRACE
  bool is_simple_trace_enabled() const;
  void enable_simple_trace();
  void disable_simple_trace();
#endif

 private:
  VERBOSE_LEVEL verbose_level;
  XPU_BACKEND xpu_backend;
  FP32_MATH_MODE fp32_math_mode;

  ENV_VAL sync_mode_enabled;
  ENV_VAL tile_as_device_enabled;
  ENV_VAL onednn_layout_enabled;

#ifdef BUILD_SIMPLE_TRACE
  ENV_VAL simple_trace_enabled;
#endif
};

} // namespace dpcpp
} // namespace xpu
