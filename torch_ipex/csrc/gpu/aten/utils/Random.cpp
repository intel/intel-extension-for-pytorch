#include <core/SYCLException.h>
#include <core/SYCLUtils.h>

#include <random>

static uint64_t createSeed(std::random_device& rd)
{
  // limit to 53 bits to ensure unique representation in double
  uint64_t seed = (((uint64_t)rd()) << 32) + rd();
  return seed & 0x1FFFFFFFFFFFFF;
}

/* Initialize generator array (must be called before any other function) */
void THSYCLRandom_init(THSYCLState* state, int devices)
{
  THSYCLRNGState* rng_state = THSYCLState_getRngState(state);
  rng_state->num_devices = devices;
  rng_state->gen = (THSYCLGenerator*)malloc(rng_state->num_devices * sizeof(THSYCLGenerator));
  std::random_device rd;
  for (int i = 0; i < rng_state->num_devices; ++i)
  {
    new (&rng_state->gen[i].mutex) std::mutex();
    rng_state->gen[i].state.initial_seed = createSeed(rd);
  }
}

/* Destroy generators and free memory */
void THSYCLRandom_shutdown(THSYCLState* state)
{
  THSYCLRNGState* rng_state = THSYCLState_getRngState(state);
  if (rng_state->gen == NULL) return;
  free(rng_state->gen);
  rng_state->gen = NULL;
}

/* Get the generator for the current device, but does not initialize the state */
static THSYCLGenerator* THSYCLRandom_rawGenerator(THSYCLState* state)
{
  THSYCLRNGState* rng_state = THSYCLState_getRngState(state);
  c10::DeviceIndex curDev = -1;
  C10_SYCL_CHECK(c10::sycl::syclGetDevice(&curDev));

  if (curDev >= rng_state->num_devices) THError("Invalid SYCL device index.");
  return &rng_state->gen[curDev];
}

/* Get the generator for the current device and initializes it if necessary */
THSYCLGenerator* THSYCLRandom_getGenerator(THSYCLState* state)
{
  THSYCLGenerator* gen = THSYCLRandom_rawGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);
  if (gen->state.initf == 0)
  {
    gen->state.initf = 1;
  }
  return gen;
}

uint64_t THSYCLRandom_seed(THSYCLState* state)
{
  std::random_device rd;
  uint64_t s = createSeed(rd);
  THSYCLRandom_manualSeed(state, s);
  return s;
}

uint64_t THSYCLRandom_seedAll(THSYCLState* state)
{
  std::random_device rd;
  uint64_t s = createSeed(rd);
  THSYCLRandom_manualSeedAll(state, s);
  return s;
}

/* Get the initial seed */
uint64_t THSYCLRandom_initialSeed(THSYCLState* state)
{
  THSYCLGenerator* gen = THSYCLRandom_getGenerator(state);
  return gen->state.initial_seed;
}

void THSYCLRandom_setRNGState(THSYCLState* state, THByteTensor *rng_state)
{
  THSYCLGenerator* gen = THSYCLRandom_getGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);

  static const size_t seed_size = sizeof(gen->state.initial_seed);
  static const size_t total_size = seed_size;
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

  memcpy(&gen->state.initial_seed, THByteTensor_data(rng_state), seed_size);
}

/* Manually set the seed */
void THSYCLRandom_manualSeed(THSYCLState* state, uint64_t seed)
{
  THSYCLGenerator* gen = THSYCLRandom_rawGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);
  gen->state.initial_seed = seed;
}

void THSYCLRandom_manualSeedAll(THSYCLState* state, uint64_t seed)
{
  THSYCLRNGState* rng_state = THSYCLState_getRngState(state);
  c10::DeviceIndex currentDevice;
  THSYCLCheck(c10::sycl::syclGetDevice(&currentDevice));
  for (int i = 0; i < rng_state->num_devices; ++i) {
    THSYCLCheck(c10::sycl::syclSetDevice(i));
    THSYCLRandom_manualSeed(state, seed);
  }
  THSYCLCheck(c10::sycl::syclSetDevice(currentDevice));
}

#include <legacy/generic/THSYCLTensorRandom.cpp>
#include <legacy/THSYCLGenerateAllTypes.h>
