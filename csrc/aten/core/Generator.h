#pragma once

#include <ATen/Utils.h>
#include <ATen/core/Generator.h>
#include <utils/Macros.h>

using namespace at;

namespace xpu {
namespace dpcpp {

struct IPEX_API DPCPPGeneratorImpl : public GeneratorImpl {
  // Constructors
  DPCPPGeneratorImpl(DeviceIndex device_index = -1);
  ~DPCPPGeneratorImpl() = default;

  // DPCPPGeneratorImpl methods
  std::shared_ptr<DPCPPGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread();
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static DeviceType device_type();
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;

 private:
  DPCPPGeneratorImpl* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
};

namespace detail {

Generator createDPCPPGenerator(DeviceIndex device_index = -1);

IPEX_API const Generator& getDefaultDPCPPGenerator(
    DeviceIndex device_index = -1);

} // namespace detail
} // namespace dpcpp
} // namespace xpu
