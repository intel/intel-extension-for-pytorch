#include <core/Functions.h>
#include <core/Generator.h>

namespace at {

namespace dpcpp {
namespace detail {

static std::once_flag num_gpu_init_flag;

static int64_t num_gpus;

static std::deque<std::once_flag> dpcpp_gens_init_flag;

static std::vector<std::shared_ptr<DPCPPGenerator>> default_gens_dpcpp;

static void initDPCPPGenVector() {
  num_gpus = device_count();
  dpcpp_gens_init_flag.resize(num_gpus);
  default_gens_dpcpp.resize(num_gpus);
}

DPCPPGenerator* getDefaultDPCPPGenerator(DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initDPCPPGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_gpus);
  }
  std::call_once(dpcpp_gens_init_flag[idx], [&] {
    default_gens_dpcpp[idx] = std::make_shared<DPCPPGenerator>(idx);
    default_gens_dpcpp[idx]->seed();
  });
  return default_gens_dpcpp[idx].get();
}

std::shared_ptr<DPCPPGenerator> createDPCPPGenerator(DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initDPCPPGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");
  auto gen = std::make_shared<DPCPPGenerator>(idx);
  gen->set_current_seed(default_rng_seed_val);
  gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail
} // namespace dpcpp

DPCPPGenerator::DPCPPGenerator(DeviceIndex device_index)
    : Generator{Device(DeviceType::DPCPP, device_index)} {}

void DPCPPGenerator::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

uint64_t DPCPPGenerator::current_seed() const {
  return seed_;
}

uint64_t DPCPPGenerator::seed() {
  auto random = detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

void DPCPPGenerator::set_philox_offset_per_thread(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

uint64_t DPCPPGenerator::philox_offset_per_thread() {
  return philox_offset_per_thread_;
}

std::pair<uint64_t, uint64_t> DPCPPGenerator::philox_engine_inputs(
    uint64_t increment) {
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

DeviceType DPCPPGenerator::device_type() {
  return DeviceType::DPCPP;
}

std::shared_ptr<DPCPPGenerator> DPCPPGenerator::clone() const {
  return std::shared_ptr<DPCPPGenerator>(this->clone_impl());
}

DPCPPGenerator* DPCPPGenerator::clone_impl() const {
  auto gen = new DPCPPGenerator(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

} // namespace at
