#include "linear_nnc.h"

#include "linear_abs.h"
#include "linear_add.h"
#include "linear_add_relu.h"
#include "linear_clamp.h"
#include "linear_elu.h"
#include "linear_exp.h"
#include "linear_gelu.h"
#include "linear_hardsigmoid.h"
#include "linear_hardswish.h"
#include "linear_leaky_relu.h"
#include "linear_log.h"
#include "linear_mish.h"
#include "linear_non.h"
#include "linear_pow.h"
#include "linear_relu.h"
#include "linear_round.h"
#include "linear_sigmoid.h"
#include "linear_silu.h"
#include "linear_sqrt.h"
#include "linear_square.h"
#include "linear_tanh.h"

#include "csrc/cpu/jit/cpu/tensorexpr/nnc_lowering_register.h"
#include "csrc/cpu/jit/cpu/tensorexpr/operator_schema.h"

namespace {

using namespace torch_ipex::jit::cpu::tensorexpr;
using NoneTrait = LoweringFuncTrait<LinearFusedOp::kLinearNone>;
using AbsTrait = LoweringFuncTrait<LinearFusedOp::kLinearAbs>;
using ExpTrait = LoweringFuncTrait<LinearFusedOp::kLinearExp>;
using HardswishTrait = LoweringFuncTrait<LinearFusedOp::kLinearHardswish>;
using MishTrait = LoweringFuncTrait<LinearFusedOp::kLinearMish>;
using SigmoidTrait = LoweringFuncTrait<LinearFusedOp::kLinearSigmoid>;
using ReluTrait = LoweringFuncTrait<LinearFusedOp::kLinearRelu>;
using SqrtTrait = LoweringFuncTrait<LinearFusedOp::kLinearSqrt>;
using SquareTrait = LoweringFuncTrait<LinearFusedOp::kLinearSquare>;
using TanhTrait = LoweringFuncTrait<LinearFusedOp::kLinearTanh>;
using SiluTrait = LoweringFuncTrait<LinearFusedOp::kLinearSilu>;
using LogTrait = LoweringFuncTrait<LinearFusedOp::kLinearLog>;
using RoundTrait = LoweringFuncTrait<LinearFusedOp::kLinearRound>;
using ClampTrait = LoweringFuncTrait<LinearFusedOp::kLinearClamp>;
using EluTrait = LoweringFuncTrait<LinearFusedOp::kLinearElu>;
using GeluTrait = LoweringFuncTrait<LinearFusedOp::kLinearGelu>;
using PowTrait = LoweringFuncTrait<LinearFusedOp::kLinearPow>;
using LeakyReluTrait = LoweringFuncTrait<LinearFusedOp::kLinearLeakyRelu>;
using AddTrait = LoweringFuncTrait<LinearFusedOp::kLinearAdd>;
using HardsigmoidTrait = LoweringFuncTrait<LinearFusedOp::kLinearHardsigmoid>;
using AddReluTrait = LoweringFuncTrait<LinearFusedOp::kLinearAddRelu>;

#define REG_NNC_OPERATOR(schema, trait)     \
  static NNCOperatorRegister schema##trait( \
      schema,                               \
      trait::get_external_func(),           \
      computeLinear<trait>,                 \
      nncLinear<trait>)

REG_NNC_OPERATOR(kLinearNoneSchema, NoneTrait);
REG_NNC_OPERATOR(kLinearAbsSchema, AbsTrait);
REG_NNC_OPERATOR(kLinearExpSchema, ExpTrait);
REG_NNC_OPERATOR(kLinearHardswishSchema, HardswishTrait);
REG_NNC_OPERATOR(kLinearMishSchema, MishTrait);
REG_NNC_OPERATOR(kLinearSigmoidSchema, SigmoidTrait);
REG_NNC_OPERATOR(kLinearReluSchema, ReluTrait);
REG_NNC_OPERATOR(kLinearSqrtSchema, SqrtTrait);
REG_NNC_OPERATOR(kLinearSquareSchema, SquareTrait);
REG_NNC_OPERATOR(kLinearTanhSchema, TanhTrait);
REG_NNC_OPERATOR(kLinearSiluSchema, SiluTrait);
REG_NNC_OPERATOR(kLinearLogSchema, LogTrait);
REG_NNC_OPERATOR(kLinearRoundSchema, RoundTrait);
REG_NNC_OPERATOR(kLinearClampSchema, ClampTrait);
REG_NNC_OPERATOR(kLinearEluSchema, EluTrait);
REG_NNC_OPERATOR(kLinearGeluSchema, GeluTrait);
REG_NNC_OPERATOR(kLinearPowSchema, PowTrait);
REG_NNC_OPERATOR(kLinearLeakyReluSchema, LeakyReluTrait);
REG_NNC_OPERATOR(kLinearAddSchema, AddTrait);
REG_NNC_OPERATOR(kLinearHardsigmoidSchema, HardsigmoidTrait);
REG_NNC_OPERATOR(kLinearAddReluSchema, AddReluTrait);
} // namespace
