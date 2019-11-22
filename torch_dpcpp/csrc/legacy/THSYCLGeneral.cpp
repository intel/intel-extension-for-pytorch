#include <THDP/THSYCLGeneral.h>
#include <TH/TH.h>
#include <THDP/THSYCLAllocator.h>
#include <THDP/THSYCLTensorRandom.h>

#include <c10/dpcpp/SYCLStream.h>

#include <stdlib.h>
#include <stdint.h>


void THSYCLState_free(THSYCLState* state)
{
  if (state != nullptr) {
    if (state->rngState != nullptr) {
      free(state->rngState);
    }
    free(state);
  }
}

THSYCLState* THSYCLState_alloc(void)
{
  THSYCLState* state = (THSYCLState*) malloc(sizeof(THSYCLState));
  memset(state, 0, sizeof(THSYCLState));
  return state;
}

void THSyclInit(THSYCLState* state)
{
  if (!state->syclDeviceAllocator) {
    state->syclDeviceAllocator = THSYCLAllocator_get();
  }

  int numDevices = 0;
  c10::sycl::syclGetDeviceCount(&numDevices);
  state->numDevices = numDevices;

  state->rngState = (THSYCLRNGState *)malloc(sizeof(THSYCLRNGState));
  THSYCLRandom_init(state, numDevices);
}

struct THSYCLRNGState* THSYCLState_getRngState(THSYCLState *state)
{
  return state->rngState;
}

c10::Allocator* THSYCLState_getSYCLHostAllocator(THSYCLState* state)
{
  return state->syclHostAllocator;
}

void __THSYCLCheck(int err, const char *file, const int line)
{
  if(err != SYCL_SUCCESS)
  {
    static int alreadyFailed = 0;
    if(!alreadyFailed) {
      fprintf(stderr, "THSYCLCheck FAIL file=%s line=%i error=%i\n", file, line, err);
      alreadyFailed = 1;
    }
    _THError(file, line, "SYCL runtime error (%d)", err);
  }
}
