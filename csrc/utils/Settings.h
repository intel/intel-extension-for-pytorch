#pragma once

#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

enum IPEX_API XPU_BACKEND { XB_GPU = 0, XB_CPU = 1, XB_AUTO = 2, XB_MAX = 3 };

class IPEX_API Settings final {
 public:
  Settings();

  static Settings& I(); // Singleton

  int get_verbose_level() const;
  void set_verbose_level(int level);

  XPU_BACKEND get_xpu_backend() const;
  void set_xpu_backend(XPU_BACKEND backend);

  bool is_xpu_sync_mode_enabled() const;
  void enable_xpu_sync_mode();
  void disable_xpu_sync_mode();

  bool is_tile_as_device_enabled() const;

  bool is_layout_opt_enabled() const;
  void enable_layout_opt();
  void disable_layout_opt();

  bool is_tf32_mode_enabled() const;
  void enable_tf32_mode();
  void disable_tf32_mode();

  bool set_onednn_verbose(int level);
  bool set_onemkl_verbose(int level);

  bool is_onemkl_enabled() const;

  bool is_channels_last_1d_enabled() const;

  bool is_double_disabled() const;

#ifdef BUILD_SIMPLE_TRACE
  bool is_simple_trace_enabled() const;
  void enable_simple_trace();
  void disable_simple_trace();
#endif

 private:
  int verbose_level;
  XPU_BACKEND xpu_backend;

  bool xpu_sync_mode_enabled;
  bool tile_as_device_enabled;
  bool layout_opt_enabled;
  bool tf32_mode_enabled;

#ifdef BUILD_SIMPLE_TRACE
  bool simple_trace_enabled;
#endif
};

} // namespace dpcpp
} // namespace xpu
