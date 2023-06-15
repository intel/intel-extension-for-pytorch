#pragma once

namespace torch_ipex {
namespace jit {
namespace cpu {
namespace tensorexpr {

extern const char kMmDivSchema[];
extern const char kConvNoneSchema[];
extern const char kConvReluSchema[];
extern const char kConvAddReluSchema[];
extern const char kConvAbsSchema[];
extern const char kConvClampSchema[];
extern const char kConvEluSchema[];
extern const char kConvExpSchema[];
extern const char kConvGeluSchema[];
extern const char kConvHardswishSchema[];
extern const char kConvLogSchema[];
extern const char kConvMishSchema[];
extern const char kConvSigmoidSchema[];
extern const char kConvPowSchema[];
extern const char kConvRoundSchema[];
extern const char kConvSqrtSchema[];
extern const char kConvSquareSchema[];
extern const char kConvTanhSchema[];
extern const char kConvLeakyReluSchema[];
extern const char kConvSiluSchema[];
extern const char kConvAddSchema[];
extern const char kConvHardsigmoidSchema[];
extern const char kLinearNoneSchema[];
extern const char kLinearAbsSchema[];
extern const char kLinearExpSchema[];
extern const char kLinearHardswishSchema[];
extern const char kLinearMishSchema[];
extern const char kLinearSigmoidSchema[];
extern const char kLinearReluSchema[];
extern const char kLinearSqrtSchema[];
extern const char kLinearSquareSchema[];
extern const char kLinearTanhSchema[];
extern const char kLinearSiluSchema[];
extern const char kLinearLogSchema[];
extern const char kLinearRoundSchema[];
extern const char kLinearClampSchema[];
extern const char kLinearEluSchema[];
extern const char kLinearGeluSchema[];
extern const char kLinearPowSchema[];
extern const char kLinearLeakyReluSchema[];
extern const char kLinearAddSchema[];
extern const char kLinearHardsigmoidSchema[];
extern const char kLinearAddReluSchema[];

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
