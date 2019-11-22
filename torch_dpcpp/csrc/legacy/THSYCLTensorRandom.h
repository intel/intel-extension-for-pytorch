#ifndef TH_SYCL_TENSOR_RANDOM_INC
#define TH_SYCL_TENSOR_RANDOM_INC

#include <THDP/THSYCLTensor.h>

#include <THDP/generic/THSYCLTensorRandom.h>
#include <THDP/THSYCLGenerateAllTypes.h>

typedef struct THSYCLGenerator THSYCLGenerator;

typedef struct THSYCLRNGState {
  /* One generator per SYCL device */
  THSYCLGenerator* gen;
  int num_devices;
} THSYCLRNGState;

struct THSYCLState;

THSYCL_API void THSYCLRandom_init(struct THSYCLState *state, int num_devices);
THSYCL_API void THSYCLRandom_shutdown(struct THSYCLState *state);
THSYCL_API uint64_t THSYCLRandom_seed(struct THSYCLState *state);
THSYCL_API uint64_t THSYCLRandom_seedAll(struct THSYCLState *state);
THSYCL_API void THSYCLRandom_manualSeed(struct THSYCLState *state, uint64_t the_seed_);
THSYCL_API void THSYCLRandom_manualSeedAll(struct THSYCLState *state, uint64_t the_seed_);
THSYCL_API uint64_t THSYCLRandom_initialSeed(THSYCLState* state);
THSYCL_API void THSYCLRandom_getRNGState(struct THSYCLState *state, THByteTensor *rng_state);
THSYCL_API void THSYCLRandom_setRNGState(struct THSYCLState *state, THByteTensor *rng_state);

#endif
