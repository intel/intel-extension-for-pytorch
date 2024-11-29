#include <runtime/CachingDeviceAllocator.h>
#include <runtime/Utils.h>
#include <runtime/XPUGraph.h>
#include <runtime/XPUGraphPhiloxGenerator.h>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/xpu/XPUFunctions.h>

#include <chrono>
#include <mutex>
#include <thread>

namespace torch_ipex::xpu::dpcpp {

using xpuGraph_t = sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::modifiable>;
using xpuGraphExec_t = sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::executable>;
using queue_state = sycl::ext::oneapi::experimental::queue_state;

inline QueueState queryQueueState(sycl::queue& queue) {
  auto state = queue.ext_oneapi_get_state();
  switch (state) {
    case queue_state::executing:
      return QueueState::Executing;
    case queue_state::recording:
      return QueueState::Recording;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown SYCL queue_state queried by graph mode");
  }
}

QueueState currentQueueState() {
  return queryQueueState(dpcppGetCurrentQueue());
}

struct XPUGraphImpl {
  XPUGraphImpl();
  ~XPUGraphImpl();

  static void inc_pending_event_queries();
  static void dec_pending_event_queries();
  static int num_pending_event_queries();
  void begin_recording(MempoolId_t pool = {0, 0});
  void end_recording();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void print_graph(const std::string& debug_path);

 protected:
  std::unique_ptr<xpuGraph_t> graph_;
  std::unique_ptr<xpuGraphExec_t> graph_exec_;

  static std::atomic<int> pending_event_queries;

  // internal states so reset() can do its best cleaning up
  // Set to true in end_recording if graph_.end_recording() succeeded
  // Set back to false soon after, when graph_ is consumed by finalize()
  // to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in end_recording if graph_.finalize() succeeded
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, used to
  // specify the pool.
  CaptureId_t id_;

  // the ID assigned by framework during graph recording,
  // used to identify when a queue is participating in recording
  CaptureId_t capture_id_ = -1;

  // uuid used to request a particular private mempool from
  // DeviceCachingAllocator. By default, this will be set to {id_, 0}.
  //
  // If begin_recording is called with "pool=other_graph.pool()", this graph's
  // mempool_id_ will be set to the other graph's mempool_id_, and therefore
  // share a mempool with the other graph.
  //
  // If begin_recording is called with "pool=handle" where "handle" came from
  // graph_pool_handle(), it will share a mempool with any other captures that
  // used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

  // Queue on which recording began
  // To record on a queue, all parameters passed in creating the graph_ are all
  // binding to this queue. Including the queue's context and its device.
  // Also, allocated private memory pool should also be binding to this queue
  // and belonging device.
  sycl::queue& recording_queue_;
  DeviceId recording_device_;

  // Default generator on device where recording began
  XPUGraphPhiloxGenerator* recording_gen_;

  // RNG state trackers
  at::Tensor seed_extragraph_;
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;
};

static bool _xpu_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

MempoolId_t graph_pool_handle() {
  // id starts at 1. 0 is reserved to mean NULL
  static std::atomic<CaptureId_t> uid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created
  // in begin_recording.
  return {0, uid++};
}

// Get the expected id of a recording sequence so that we can call
// beginAllocateQueueToPool before starting a graph recording
CaptureId_t capture_sequence_id() {
  // id starts at 1. 0 is reserved to mean NULL
  static std::atomic<CaptureId_t> uuid{1};
  return uuid++;
}

std::atomic<int> XPUGraphImpl::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a oneCCL
// watchdog so that they can be resolved before the recording begins. Note that
// event queries are not allowed during a graph recording.
void XPUGraphImpl::inc_pending_event_queries() {
  pending_event_queries++;
}

void XPUGraphImpl::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(
      pending_event_queries > 0,
      "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int XPUGraphImpl::num_pending_event_queries() {
  return pending_event_queries;
}

XPUGraphImpl::XPUGraphImpl() : recording_queue_(dpcppGetCurrentQueue()) {}

void XPUGraphImpl::begin_recording(MempoolId_t pool /*=0*/) {
  TORCH_CHECK(
      !has_graph_exec_,
      "This XPUGraph instance already owns a recorded graph. "
      "To record a new graph, create a new instance.");

  // For now, a XPUGraph instance only accommodates the default generator on the
  // device that's current when recording begins. If any op in the captured
  // region uses a non-default generator, or a generator on another device, the
  // offending generator will throw an error. These restrictions simplify
  // XPUGraph, but could be relaxed in the future: in principle, the underlying
  // XPU calls do permit cross-device ops to be captured.
  auto* gen = get_generator_or_default<at::XPUGeneratorImpl>(
      c10::nullopt, at::xpu::detail::getDefaultXPUGenerator());
  recording_gen_ = new XPUGraphPhiloxGenerator(gen);

  auto options = TensorOptions().device(at::kXPU).dtype(at::kLong);
  seed_extragraph_ = at::empty({1}, options);
  offset_extragraph_ = at::empty({1}, options);

  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  recording_gen_->graph_recording_prologue(
      seed_extragraph_.data_ptr<int64_t>(),
      offset_extragraph_.mutable_data_ptr<int64_t>());

  recording_queue_ = dpcppGetCurrentQueue();
  // Notice: currently the recording queue should always be the one run on
  // current device and the current device id can be recorded directly
  // even without querying from the sycl queue object. But this is not
  // safe and strong at the design layer.
  recording_device_ = at::xpu::current_device();

  id_ = capture_sequence_id();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's begin_recording, first should be
    // nonzero. If pool was created by graph_pool_handle, second should be
    // nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our
    // mempool_id_. Sets just the first value, to distinguish it from
    // MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  CachingDeviceAllocator::Instance()->beginAllocateToPool(
      recording_device_, mempool_id_, [this](sycl::queue* queue) {
        return (queryQueueState(this->recording_queue_) ==
                QueueState::Recording) &&
            (&this->recording_queue_ == queue);
      });

  // At this point, any oneCCL watchdogs should be aware that we are in
  // recording mode and therefore should not enqueue any additional work that
  // could be event-queried. We still must wait on any existing work that has
  // not been cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE(
        "Waiting for pending oneCCL work to finish before starting graph capture.");
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  auto g =
      xpuGraph_t(recording_queue_.get_context(), recording_queue_.get_device());
  graph_ = std::make_unique<xpuGraph_t>(g);
  graph_->begin_recording(recording_queue_);

  TORCH_INTERNAL_ASSERT(currentQueueState() == QueueState::Recording);

  TORCH_INTERNAL_ASSERT(id_ > 0);
}

void XPUGraphImpl::end_recording() {
  TORCH_CHECK(
      dpcppGetCurrentQueue() == recording_queue_,
      "Recording must end on the same queue it began on.");

  graph_->end_recording();

  CachingDeviceAllocator::Instance()->endAllocateToPool(
      recording_device_, mempool_id_);

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");
  has_graph_ = true;

  auto ge = graph_->finalize();
  graph_exec_ = std::make_unique<xpuGraphExec_t>(ge);
  TORCH_CHECK(graph_exec_ != nullptr, "Invalid executable graph.");
  has_graph_exec_ = true;

  auto* gen = get_generator_or_default<at::XPUGeneratorImpl>(
      c10::nullopt, at::xpu::detail::getDefaultXPUGenerator());
  TORCH_CHECK(
      gen == recording_gen_->get_generator(),
      "Default XPU RNG generator on current device at recording end "
      "is different from default generator on current device "
      "when recording began");
  wholegraph_increment_ = recording_gen_->graph_recording_epilogue();

#if defined(SYCL_EXT_ONEAPI_GRAPH_FUSION)
  size_t numXPUGraphNodes = graph_->get_nodes().size();
  if (numXPUGraphNodes == 0) {
    TORCH_WARN(
        "The XPU Graph is empty. This usually means that the graph was ",
        "attempted to be recorded on wrong device or queue.");
  }
#endif

  // check if debug path is set
  if (!_xpu_graphs_debug) {
    // Now that we've instantiated graph_ into graph_exec_,
    // we don't need graph_ anymore.
    delete graph_.release();
    has_graph_ = false;
  } else {
    TORCH_WARN(
        "DEBUG: TORCH_XPUGRAPHS_DEBUG_PATH detected. graph_ will not be freed until print_graph is called.");
  }
}

void XPUGraphImpl::replay() {
  TORCH_CHECK(
      has_graph_exec_,
      "Called XPUGraphImpl::replay without a preceeding successful capture.");

  // FIXME:
  // To create the device guard, we need the index of the corresponding device
  // which is binding to the recording queue. But the XPUDevice.h doesn't offer
  // such an API for querying the device index from sycl queue or sycl device
  // object. So as having to walk around, here should construct a dummy address
  // to get a pointer which can be used to query the device index via given API.
  // Please fix this in the future upstream work.
  void* ptr = sycl::malloc_device(
      0, recording_queue_.get_device(), c10::xpu::get_device_context());
  c10::OptionalDeviceGuard device_guard{
      c10::Device(c10::DeviceType::XPU, recording_device_)};

  // Just like any RNG consumer kernel!
  auto* gen = get_generator_or_default<at::XPUGeneratorImpl>(
      c10::nullopt, at::xpu::detail::getDefaultXPUGenerator());
  PhiloxXPUState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = recording_gen_->philox_xpu_state(wholegraph_increment_);
  }
  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));

  auto& q = dpcppGetCurrentQueue();
  q.ext_oneapi_graph(*graph_exec_);
}

void XPUGraphImpl::enable_debug_mode() {
  _xpu_graphs_debug = true;
}

void XPUGraphImpl::print_graph(const std::string& debug_path) {
#if _GLIBCXX_USE_CXX11_ABI
  if (_xpu_graphs_debug) {
    TORCH_WARN("DEBUG: calling print_graph()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling print_graph(verbose=1) to ", debug_path);
      graph_->print_graph(debug_path, /* verbose = */ 1);
    }
  } else {
    TORCH_WARN(
        "XPU Graphs debug not enabled, set with intel_extension_for_pytorch.xpu._C._xpu_enable_graphs_debug_mode");
  }
#else
  TORCH_INTERNAL_ASSERT(
      false, "XPU graph print is not supported by ABI=0 build");
#endif
}

void XPUGraphImpl::reset() {
  if (has_graph_ || has_graph_exec_) {
    CachingDeviceAllocator::Instance()->releasePool(
        recording_device_, mempool_id_);
  }
  if (has_graph_) {
    delete graph_.release();
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    delete graph_exec_.release();
    has_graph_exec_ = false;
  }
}

// Returns an id another graph's begin_recording can use to share the same
// memory pool as this graph.
MempoolId_t XPUGraphImpl::pool() {
  TORCH_CHECK(
      has_graph_exec_,
      "Called XPUGraphImpl::pool() without a preceding successful recording.");
  return mempool_id_;
}

XPUGraphImpl::~XPUGraphImpl() {
  reset();
}

//================ XPUGraph Export Interfaces ===============//
XPUGraph::XPUGraph() : p_xpu_graph(std::make_unique<XPUGraphImpl>()) {}
XPUGraph::~XPUGraph() = default;

void XPUGraph::begin_recording(MempoolId_t pool) {
  p_xpu_graph->begin_recording(pool);
}

void XPUGraph::end_recording() {
  p_xpu_graph->end_recording();
}

void XPUGraph::replay() {
  p_xpu_graph->replay();
}

void XPUGraph::reset() {
  p_xpu_graph->reset();
}

MempoolId_t XPUGraph::pool() {
  return p_xpu_graph->pool();
}

void XPUGraph::enable_debug_mode() {
  p_xpu_graph->enable_debug_mode();
}

void XPUGraph::print_graph(const std::string& debug_path) {
  p_xpu_graph->print_graph(debug_path);
}

} // namespace torch_ipex::xpu::dpcpp
