#include "conv_nnc.h"

#include "conv_abs.h"
#include "conv_add.h"
#include "conv_add_relu.h"
#include "conv_clamp.h"
#include "conv_elu.h"
#include "conv_exp.h"
#include "conv_gelu.h"
#include "conv_hardsigmoid.h"
#include "conv_hardswish.h"
#include "conv_leaky_relu.h"
#include "conv_log.h"
#include "conv_mish.h"
#include "conv_non.h"
#include "conv_pow.h"
#include "conv_relu.h"
#include "conv_round.h"
#include "conv_sigmoid.h"
#include "conv_silu.h"
#include "conv_sqrt.h"
#include "conv_square.h"
#include "conv_tanh.h"

#include "csrc/jit/cpu/tensorexpr/nnc_lowering_register.h"
#include "csrc/jit/cpu/tensorexpr/operator_schema.h"

namespace {

using namespace torch_ipex::jit::cpu::tensorexpr;
using NoneTrait = LoweringFuncTrait<ConvFusedOp::kConvNone>;
using ReLUTrait = LoweringFuncTrait<ConvFusedOp::kConvReLU>;
using AddReLUTrait = LoweringFuncTrait<ConvFusedOp::kConvAddReLU>;
using AbsTrait = LoweringFuncTrait<ConvFusedOp::kConvAbs>;
using ClampTrait = LoweringFuncTrait<ConvFusedOp::kConvClamp>;
using EluTrait = LoweringFuncTrait<ConvFusedOp::kConvElu>;
using ExpTrait = LoweringFuncTrait<ConvFusedOp::kConvExp>;
using GeluTrait = LoweringFuncTrait<ConvFusedOp::kConvGelu>;
using HardswishTrait = LoweringFuncTrait<ConvFusedOp::kConvHardswish>;
using LogTrait = LoweringFuncTrait<ConvFusedOp::kConvLog>;
using MishTrait = LoweringFuncTrait<ConvFusedOp::kConvMish>;
using SigmoidTrait = LoweringFuncTrait<ConvFusedOp::kConvSigmoid>;
using PowTrait = LoweringFuncTrait<ConvFusedOp::kConvPow>;
using RoundTrait = LoweringFuncTrait<ConvFusedOp::kConvRound>;
using SqrtTrait = LoweringFuncTrait<ConvFusedOp::kConvSqrt>;
using SquareTrait = LoweringFuncTrait<ConvFusedOp::kConvSquare>;
using TanhTrait = LoweringFuncTrait<ConvFusedOp::kConvTanh>;
using LeakyReluTrait = LoweringFuncTrait<ConvFusedOp::kConvLeakyRelu>;
using SiluTrait = LoweringFuncTrait<ConvFusedOp::kConvSilu>;
using AddTrait = LoweringFuncTrait<ConvFusedOp::kConvAdd>;
using HardsigmoidTrait = LoweringFuncTrait<ConvFusedOp::kConvHardsigmoid>;

#define REG_NNC_OPERATOR(schema, trait)     \
  static NNCOperatorRegister schema##trait( \
      schema, trait::get_external_func(), computeConv<trait>, nncConv<trait>)

REG_NNC_OPERATOR(kConvNoneSchema, NoneTrait);
REG_NNC_OPERATOR(kConvReluSchema, ReLUTrait);
REG_NNC_OPERATOR(kConvAddReluSchema, AddReLUTrait);
REG_NNC_OPERATOR(kConvAbsSchema, AbsTrait);
REG_NNC_OPERATOR(kConvClampSchema, ClampTrait);
REG_NNC_OPERATOR(kConvEluSchema, EluTrait);
REG_NNC_OPERATOR(kConvExpSchema, ExpTrait);
REG_NNC_OPERATOR(kConvGeluSchema, GeluTrait);
REG_NNC_OPERATOR(kConvHardswishSchema, HardswishTrait);
REG_NNC_OPERATOR(kConvLogSchema, LogTrait);
REG_NNC_OPERATOR(kConvMishSchema, MishTrait);
REG_NNC_OPERATOR(kConvSigmoidSchema, SigmoidTrait);
REG_NNC_OPERATOR(kConvPowSchema, PowTrait);
REG_NNC_OPERATOR(kConvRoundSchema, RoundTrait);
REG_NNC_OPERATOR(kConvSqrtSchema, SqrtTrait);
REG_NNC_OPERATOR(kConvSquareSchema, SquareTrait);
REG_NNC_OPERATOR(kConvTanhSchema, TanhTrait);
REG_NNC_OPERATOR(kConvLeakyReluSchema, LeakyReluTrait);
REG_NNC_OPERATOR(kConvSiluSchema, SiluTrait);
REG_NNC_OPERATOR(kConvAddSchema, AddTrait);
REG_NNC_OPERATOR(kConvHardsigmoidSchema, HardsigmoidTrait);
} // namespace
