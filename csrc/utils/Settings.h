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

  int get_warning_level() const;
  void set_warning_level(int level);

  XPU_BACKEND get_xpu_backend() const;
  void set_xpu_backend(XPU_BACKEND backend);

  bool is_force_sync_exec() const;
  void enable_force_sync_exec();
  void disable_force_sync_exec();

  bool is_event_profiling_enabled() const;
  void enable_event_profiling();
  void disable_event_profiling();

  bool is_tile_partition_enabled() const;
  void enable_tile_partition();
  void disable_tile_partition();

  bool is_onednn_layout_enabled() const;
  void enable_onednn_layout();
  void disable_onednn_layout();

  bool is_tf32_mode_enabled() const;
  void enable_tf32_mode();
  void disable_tf32_mode();

  bool set_onednn_verbose(int level);

  bool is_onedpl_enabled() const;

  bool is_onemkl_enabled() const;

  bool is_double_disabled() const;

 private:
  int verbose_level;
  int warning_level;
  XPU_BACKEND xpu_backend;

  bool force_sync_exec_enabled;
  bool event_profiling_enabled;
  bool tile_partition_enabled;
  bool onednn_layout_enabled;
  bool tf32_mode_enabled;
};

} // namespace dpcpp
} // namespace xpu
