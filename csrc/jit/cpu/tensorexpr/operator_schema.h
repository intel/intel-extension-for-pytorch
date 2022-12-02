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

} // namespace tensorexpr
} // namespace cpu
} // namespace jit
} // namespace torch_ipex
