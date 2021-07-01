#include <utils/Settings.h>

namespace xpu {
namespace dpcpp {

int Settings::get_verbose_level() const {
  return verbose_level;
}

void Settings::set_verbose_level(int level) {
  verbose_level = level;
}

XPU_BACKEND Settings::get_xpu_backend() const {
  return xpu_backend;
}

void Settings::set_xpu_backend(XPU_BACKEND backend) {
  xpu_backend = backend;
}

bool Settings::is_force_sync_exec() const {
  return force_sync_exec_enabled;
}

void Settings::enable_force_sync_exec(bool force_sync) {
  force_sync_exec_enabled = true;
}

void Settings::disable_force_sync_exec(bool force_sync) {
  force_sync_exec_enabled = false;
}

bool Settings::is_event_profiling_enabled() const {
  return event_profiling_enabled;
}

void Settings::enable_event_profiling() {
  event_profiling_enabled = true;
}

void Settings::disable_event_profiling() {
  event_profiling_enabled = false;
}

bool Settings::is_tile_partition_enabled() const {
  return tile_partition_enabled;
}

void Settings::enable_tile_partition() {
  tile_partition_enabled = true;
}

void Settings::disable_tile_partition() {
  tile_partition_enabled = false;
}

bool Settings::is_onednn_layout_enabled() const {
  return onednn_layout_enabled;
}

void Settings::enable_onednn_layout() {
  onednn_layout_enabled = true;
}

void Settings::disable_onednn_layout() {
  onednn_layout_enabled = false;
}

bool Settings::is_tf32_mode_enabled() const {
  return tf32_mode_enabled;
}

void Settings::enable_tf32_mode() {
  tf32_mode_enabled = true;
}

void Settings::disable_tf32_mode() {
  tf32_mode_enabled = false;
}

} // namespace dpcpp
} // namespace xpu
