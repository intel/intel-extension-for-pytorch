#pragma once

#include <utils/Macros.h>

#include <memory>

namespace torch_ipex::xpu {

namespace dpcpp {

using CaptureId_t = unsigned long long;
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

enum class QueueState {
  Executing = 0,
  Recording = 1,
};

QueueState IPEX_API currentQueueState();

// Standalone way to get a unique mempool id usable as a pool=... argument
// to XPUGraph::begin_recording
MempoolId_t IPEX_API graph_pool_handle();

struct XPUGraphImpl;

struct IPEX_API XPUGraph {
  XPUGraph();
  ~XPUGraph();

  void begin_recording(MempoolId_t pool = {0, 0});
  void end_recording();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void print_graph(const std::string& debug_path);

 private:
  std::unique_ptr<XPUGraphImpl> p_xpu_graph;
};

} // namespace dpcpp
} // namespace torch_ipex::xpu
