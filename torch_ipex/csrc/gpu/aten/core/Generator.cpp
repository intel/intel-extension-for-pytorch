#include <core/Generator.h>
#include <core/Functions.h>


namespace at {

namespace sycl { namespace detail {

static std::once_flag num_gpu_init_flag;

static int64_t num_gpus;

static std::deque<std::once_flag> sycl_gens_init_flag;

static std::vector<std::shared_ptr<SYCLGenerator>> default_gens_sycl;

static void initSYCLGenVector(){
  num_gpus = c10::sycl::device_count();
  sycl_gens_init_flag.resize(num_gpus);
  default_gens_sycl.resize(num_gpus);
}

SYCLGenerator* getDefaultSYCLGenerator(DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initSYCLGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::sycl::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_gpus);
  }
  std::call_once(sycl_gens_init_flag[idx], [&] {
    default_gens_sycl[idx] = std::make_shared<SYCLGenerator>(idx);
    default_gens_sycl[idx]->seed();
  });
  return default_gens_sycl[idx].get();
}

std::shared_ptr<SYCLGenerator> createSYCLGenerator(DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initSYCLGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::sycl::current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");
  auto gen = std::make_shared<SYCLGenerator>(idx);
  gen->set_current_seed(default_rng_seed_val);
  gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail
} // namespace sycl

SYCLGenerator::SYCLGenerator(DeviceIndex device_index)
  : Generator{Device(DeviceType::DPCPP, device_index)} { }

void SYCLGenerator::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

uint64_t SYCLGenerator::current_seed() const {
  return seed_;
}

uint64_t SYCLGenerator::seed() {
  auto random = detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

void SYCLGenerator::set_philox_offset_per_thread(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

uint64_t SYCLGenerator::philox_offset_per_thread() {
  return philox_offset_per_thread_;
}

std::pair<uint64_t, uint64_t> SYCLGenerator::philox_engine_inputs(uint64_t increment) {
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

DeviceType SYCLGenerator::device_type() {
  return DeviceType::DPCPP;
}

std::shared_ptr<SYCLGenerator> SYCLGenerator::clone() const {
  return std::shared_ptr<SYCLGenerator>(this->clone_impl());
}

SYCLGenerator* SYCLGenerator::clone_impl() const {
  auto gen = new SYCLGenerator(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

} // namespace at
