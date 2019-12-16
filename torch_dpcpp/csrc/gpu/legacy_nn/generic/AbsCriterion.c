#ifndef THSYCL_GENERIC_FILE
#define THSYCL_GENERIC_FILE "legacy_nn/generic/AbsCriterion.c"
#else

void THNN_(AbsCriterion_updateOutput)(
          THSYCLState *state,
          THSYCLTensor *input,
          THSYCLTensor *target,
          THSYCLTensor *output,
          int64_t reduction)
{
  throw std::runtime_error("THNN_(AbsCriterion_updateOutput)() not implemented");
}

void THNN_(AbsCriterion_updateGradInput)(
          THSYCLState *state,
          THSYCLTensor *input,
          THSYCLTensor *target,
          THSYCLTensor *gradOutput,
          THSYCLTensor *gradInput,
          int64_t reduction)
{
  throw std::runtime_error("THNN_(AbsCriterion_updateGradInput)() not implemented");
}
#endif
