#include <runtime/XPUGraph.h>
#include <runtime/XPUGraphPhiloxGenerator.h>

namespace torch_ipex::xpu {

namespace dpcpp {

XPUGraphPhiloxGenerator::XPUGraphPhiloxGenerator(at::XPUGeneratorImpl* gen)
    : correlated_gen_(gen){};

/**
 * Called by XPUGraph to prepare this instance for a graph recording region.
 * offset_extragraph is the initial offset at the start of the graphed region.
 * offset_intragraph tracks the offset in the graphed region.
 */
void XPUGraphPhiloxGenerator::graph_recording_prologue(
    int64_t* seed_extragraph,
    int64_t* offset_extragraph) {
  seed_extragraph_ = seed_extragraph;
  offset_extragraph_ = offset_extragraph;
  offset_intragraph_ = 0;
  graph_expects_this_gen_ = true;
}

/**
 * Called by XPUGraph to finalize a graph capture region for this instance.
 */
uint64_t XPUGraphPhiloxGenerator::graph_recording_epilogue() {
  graph_expects_this_gen_ = false;
  return offset_intragraph_;
}

/**
 * Gets the seed and philox offset value to be used in
 * curandStatePhilox4_32_10, in an opaque PhiloxXPUState that's safe
 * and can be used non-divergently in callers whether XPU graph
 * recording is underway or not.
 *
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate.
 *
 * Increment should be at least the number of curand() random numbers used in
 * each thread. It is the user's responsibility to make sure the increment
 * for philox is never smaller than the number of curand() calls. Increment
 * value > the number of curand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 */
PhiloxXPUState XPUGraphPhiloxGenerator::philox_xpu_state(uint64_t increment) {
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  if (currentQueueState() == QueueState::Recording) {
    TORCH_CHECK(
        graph_expects_this_gen_,
        "philox_xpu_state for an unexpected XPU generator used during recording. ");
    // also enforce RNG offset % 4 == 0 in a CUDA-like way
    TORCH_INTERNAL_ASSERT(this->offset_intragraph_ % 4 == 0);
    uint32_t offset = this->offset_intragraph_;
    TORCH_INTERNAL_ASSERT(
        this->offset_intragraph_ <=
        std::numeric_limits<uint32_t>::max() - increment);
    this->offset_intragraph_ += increment;
    return PhiloxXPUState(
        this->seed_extragraph_, this->offset_extragraph_, offset);
  } else {
    TORCH_CHECK(
        !graph_expects_this_gen_,
        "XPU generator expects graph recording to be underway, "
        "but the current queue is not recording.");
    uint64_t offset = this->correlated_gen_->philox_offset_per_thread();
    TORCH_INTERNAL_ASSERT(offset % 4 == 0);
    this->correlated_gen_->set_philox_offset_per_thread(offset + increment);
    return PhiloxXPUState(this->correlated_gen_->seed(), offset);
  }
}

} // namespace dpcpp
} // namespace torch_ipex::xpu
