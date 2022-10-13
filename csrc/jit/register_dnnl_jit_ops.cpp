#include <ATen/native/quantized/PackedParams.h>
#include <quantized/QUtil.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <iostream>
#include <utility>
#include <vector>
#include "ATen/core/ivalue.h"
#include "ATen/core/stack.h"
#include <ATen/native/quantized/PackedParams.h>
#include "accelerated_ops.h"
#include "c10/core/Scalar.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "c10/util/Metaprogramming.h"
#include "c10/util/Optional.h"
#include "c10/util/TypeList.h"
#include "c10/util/intrusive_ptr.h"
#include "dpcpp_ops.h"

namespace {
// using namespace torch;
using namespace at::native;
using namespace torch::jit;
template <typename T>
struct IvalueConvert {
  T operator()(c10::IValue&) {}
};

/* DEFINE CONVERTER
DEFINE CONVERTER is used to generate template specialization of
IvalueConvert<T>, all the parameter type in schema should have its
IvalueConverter specialize version, otherwise error might occur during
compilation. Through this macro, any T type will generate its T, T&, const T,
const T& specialization.
*/
#define DEFINE_CONVERTER(T, mf)           \
  template <>                             \
  struct IvalueConvert<T> {               \
    T operator()(c10::IValue& i_value) {  \
      return i_value.mf;                  \
    }                                     \
    T operator()(c10::IValue&& i_value) { \
      return i_value.mf;                  \
    }                                     \
  };                                      \
  template <>                             \
  struct IvalueConvert<T&> {              \
    T operator()(c10::IValue& i_value) {  \
      val_ = i_value.mf;                  \
      return val_;                        \
    }                                     \
    T operator()(c10::IValue&& i_value) { \
      val_ = i_value.mf;                  \
      return val_;                        \
    }                                     \
    T val_;                               \
  };                                      \
  template <>                             \
  struct IvalueConvert<const T> {         \
    T operator()(c10::IValue& i_value) {  \
      return i_value.mf;                  \
    }                                     \
    T operator()(c10::IValue&& i_value) { \
      return i_value.mf;                  \
    }                                     \
  };                                      \
  template <>                             \
  struct IvalueConvert<const T&> {        \
    T operator()(c10::IValue& i_value) {  \
      val_ = i_value.mf;                  \
      return val_;                        \
    }                                     \
    T operator()(c10::IValue&& i_value) { \
      val_ = i_value.mf;                  \
      return val_;                        \
    }                                     \
    T val_;                               \
  };

#define DEFINE_CUSTOMTYPE_CONVERT(T)                          \
  template <>                                                 \
  struct IvalueConvert<T> {                                   \
    c10::intrusive_ptr<T> operator()(c10::IValue& i_value) {  \
      return i_value.toCustomClass<T>();                      \
    }                                                         \
    c10::intrusive_ptr<T> operator()(c10::IValue&& i_value) { \
      return i_value.toCustomClass<T>();                      \
    }                                                         \
  };

#define DEFINE_OPTIONAL_CONVERTER(T, mf)          \
  template <>                                     \
  struct IvalueConvert<c10::optional<T>> {        \
    T operator()(IValue& i_value) {               \
      if (i_value.isNone()) {                     \
        T ret;                                    \
        return ret;                               \
      } else {                                    \
        return i_value.mf;                        \
      }                                           \
    }                                             \
    T operator()(IValue&& i_value) {              \
      if (i_value.isNone()) {                     \
        T ret;                                    \
        return ret;                               \
      } else {                                    \
        return i_value.mf;                        \
      }                                           \
    }                                             \
  };                                              \
  template <>                                     \
  struct IvalueConvert<const c10::optional<T>> {  \
    T operator()(IValue& i_value) {               \
      if (i_value.isNone()) {                     \
        T ret;                                    \
        return ret;                               \
      } else {                                    \
        return i_value.mf;                        \
      }                                           \
    }                                             \
    T operator()(IValue&& i_value) {              \
      if (i_value.isNone()) {                     \
        T ret;                                    \
        return ret;                               \
      } else {                                    \
        return i_value.mf;                        \
      }                                           \
    }                                             \
  };                                              \
  template <>                                     \
  struct IvalueConvert<c10::optional<T>&> {       \
    T operator()(IValue& i_value) {               \
      if (i_value.isNone()) {                     \
        T ret;                                    \
        return ret;                               \
      } else {                                    \
        return i_value.mf;                        \
      }                                           \
    }                                             \
    T operator()(IValue&& i_value) {              \
      if (i_value.isNone()) {                     \
        T ret;                                    \
        return ret;                               \
      } else {                                    \
        return i_value.mf;                        \
      }                                           \
    }                                             \
  };                                              \
  template <>                                     \
  struct IvalueConvert<const c10::optional<T>&> { \
    T operator()(IValue& i_value) {               \
      if (i_value.isNone()) {                     \
        T ret;                                    \
        return ret;                               \
      } else {                                    \
        return i_value.mf;                        \
      }                                           \
    }                                             \
    T operator()(IValue&& i_value) {              \
      if (i_value.isNone()) {                     \
        T ret;                                    \
        return ret;                               \
      } else {                                    \
        return i_value.mf;                        \
      }                                           \
    }                                             \
  };

DEFINE_CONVERTER(at::Tensor, toTensor())
DEFINE_CONVERTER(std::vector<int64_t>, toIntVector())
// DEFINE_CONVERTER(c10::ArrayRef<long>, toIntVector())
DEFINE_CONVERTER(at::Scalar, toScalar())
DEFINE_CONVERTER(at::ScalarType, toScalarType())
DEFINE_CONVERTER(at::MemoryFormat, toMemoryFormat())
DEFINE_CONVERTER(at::QScheme, toQScheme())
DEFINE_CONVERTER(c10::Device, toDevice())
DEFINE_CONVERTER(c10::intrusive_ptr<ivalue::Object>, toObject())
DEFINE_CONVERTER(c10::Stream, toStream())
DEFINE_CONVERTER(c10::Layout, toLayout())
DEFINE_CONVERTER(int, toInt())
DEFINE_CONVERTER(long, toInt())
DEFINE_CONVERTER(float, toDouble())
DEFINE_CONVERTER(double, toDouble())
DEFINE_CONVERTER(bool, toBool())
DEFINE_CONVERTER(c10::complex<double>, toComplexDouble())
DEFINE_CONVERTER(at::Dimname, toDimname())
DEFINE_CUSTOMTYPE_CONVERT(ConvPackedParamsBase<2>)

DEFINE_OPTIONAL_CONVERTER(at::Tensor, toTensor())

template <>
struct IvalueConvert<const c10::intrusive_ptr<ConvPackedParamsBase<2>>&> {
  c10::intrusive_ptr<ConvPackedParamsBase<2>> operator()(IValue& i_value) {
    return i_value.toCustomClass<ConvPackedParamsBase<2>>();
  }
  c10::intrusive_ptr<ConvPackedParamsBase<2>> operator()(IValue&& i_value) {
    return i_value.toCustomClass<ConvPackedParamsBase<2>>();
  }
};

template <>
struct IvalueConvert<at::IntArrayRef> {
  std::vector<int64_t> operator()(IValue& i_value) {
    std::vector<int64_t> ref = i_value.toIntVector();
    return ref;
  }
  std::vector<int64_t> operator()(IValue&& i_value) {
    std::vector<int64_t> ref = i_value.toIntVector();
    return ref;
  }
};

template <typename... Args>
constexpr int typeCounter() {
  return sizeof...(Args);
}

template <int n, typename Func, typename TypeList, size_t... N>
torch::jit::Operation generateJitOperation(
    Func func,
    std::string str,
    std::index_sequence<N...> seq) {
  return [=](Stack& stack) {
    auto result = torch::jit::xpu::JitFusionProxy<Func>()(
        func,
        str,
        IvalueConvert<c10::guts::typelist::element_t<N, TypeList>>()(
            std::move(peek(stack, N, n)))...);
    drop(stack, n);
    pack(stack, std::move(result));
  };
}

/* IPEX_JIT_OP_REGISTER useage:
This macro is wroten as a basic fusion op registeration method, which will
bound the schema and implementation function together as an Operator. The fused
one will be called at jit runtime as single operator rather than two. Op fusion
can bring effective improvement to jit runtime by reduce the read and write to
global memory

A standard usege looks like:

      IPEX_JIT_OP_REGISTER(fusion-schema, function-name, function-pointer)

Parameters:
-----------
    1. fusion-schema: the schema string after op fusion.

    2. function-name: function name of the fusion implementation.

    3. function-pointer: the implementation function pointer of op fusion. This
    function pointer should have exactly same signiture against fusion-schema.
It is worth to note that all the input parameter type should be specialized by
    IvalueConvert.

for example, if we have convolution and relu adjacent at the jit graph like
below:

      graph(%self : __torch__.test_fusion.Conv2dRelu, %x.1 : Tensor, %a :
Tensor): %14 : int[] = prim::Constant[value=[0, 0]]() %13 : int[] =
prim::Constant[value=[1, 1]]() %3 : int = prim::Constant[value=1]() %conv :
__torch__.torch.nn.modules.conv.Conv2d = prim::GetAttr[name="conv"](%self)
%weight : Tensor = prim::GetAttr[name="weight"](%conv) %bias : Tensor =
      prim::GetAttr[name="bias"](%conv) %11 : Tensor = aten::conv2d(%x.1,
%weight, %bias, %13, %14, %13, %3) %result.3 : Tensor = aten::relu(%11) return
      (%result.3)

we can register convolution and relu as one operator to reduce the unnecessary
global memory access. And the registeration can write as:

  IPEX_JIT_OP_REGISTER(
    "xpu::conv2d_relu(Tensor input, Tensor weight, Tensor? bias, int[2] stride,
  int[2] padding, int[2] dilation, int groups) -> Tensor", "convolution_relu",
    at::AtenIpexTypeXPU::convolution_relu)

after registeration, the jit ir graph will be like:

      graph(%self : __torch__.test_fusion.Conv2dRelu, %x.1 : Tensor, %a :
Tensor): %14 : int[] = prim::Constant[value=[0, 0]]() %13 : int[] =
prim::Constant[value=[1, 1]]() %3 : int = prim::Constant[value=1]() %conv :
__torch__.torch.nn.modules.conv.Conv2d = prim::GetAttr[name="conv"](%self)
%weight : Tensor = prim::GetAttr[name="weight"](%conv) %bias : Tensor =
        prim::GetAttr[name="bias"](%conv) %16 : Tensor = xpu::conv2d_relu(%x.1,
    %weight, %bias, %13, %14, %13, %3) return (%16)
*/

#define IPEX_JIT_OP_REGISTER(schema, name, func)                       \
  Operator(                                                            \
      schema,                                                          \
      [](const Node* node) -> Operation {                              \
        constexpr int n = c10::guts::infer_function_traits_t<decltype( \
            func)>::number_of_parameters;                              \
        return generateJitOperation<                                   \
            n,                                                         \
            decltype(func),                                            \
            c10::guts::infer_function_traits_t<decltype(               \
                func)>::parameter_types>(                              \
            func, name, std::make_index_sequence<n>());                \
      },                                                               \
      aliasAnalysisFromSchema())

#define IPEX_SUFFIX ") -> Tensor"
#define QCONV2D_PREFIX "xpu::q_conv2d_"
#define CONV2D_PREFIX "xpu::conv2d_"
#define _CONV_PREFIX "xpu::_convolution_"
#define QCONV2D_MAIN_SCHEMA \
  "(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zpoint"
#define CONV2D_MAIN_SCHEMA \
  "(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups"
#define _CONV_MAIN_SCHEMA \
  "(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transpose, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32"
#define _CONV_SUFFIX ""
#define EXTRA_SCHEMA(str) ", " str

#define IPEX_QCONV2D_JIT_OP_REGISTER(func, ...)                         \
  IPEX_JIT_OP_REGISTER(                                                 \
      QCONV2D_PREFIX #func QCONV2D_MAIN_SCHEMA __VA_ARGS__ IPEX_SUFFIX, \
      "q_conv2d_" #func,                                                \
      at::AtenIpexTypeXPU::q_conv2d_##func)

#define IPEX_CONV2D_JIT_OP_REGISTER(func, ...)                        \
  IPEX_JIT_OP_REGISTER(                                               \
      CONV2D_PREFIX #func CONV2D_MAIN_SCHEMA __VA_ARGS__ IPEX_SUFFIX, \
      "convolution_" #func,                                           \
      at::AtenIpexTypeXPU::convolution_##func)

#define IPEX__CONV_JIT_OP_REGISTER(func, ...)                       \
  IPEX_JIT_OP_REGISTER(                                             \
      _CONV_PREFIX #func _CONV_MAIN_SCHEMA __VA_ARGS__ IPEX_SUFFIX, \
      "_convolution_" #func,                                        \
      at::AtenIpexTypeXPU::_convolution_##func)

#define IPEX_LINEAR_PREFIX "xpu::linear_"
#define IPEX_LINEAR_MAIN_SCHEMA "(Tensor input, Tensor weight, Tensor? bias"

#define IPEX_LINEAR_JIT_OP_REGISTER(func, ...)                     \
  IPEX_JIT_OP_REGISTER(                                            \
      IPEX_LINEAR_PREFIX #func IPEX_LINEAR_MAIN_SCHEMA __VA_ARGS__ \
          IPEX_SUFFIX,                                             \
      "linear_" #func,                                             \
      at::AtenIpexTypeXPU::linear_##func)

/* IPEX_GENERAL_CONV2D_JIT_OP_REGISTER usage:
We decalre this macro to enable a more simple way to register the fusion pattern
of Convolution + Post-ops in jit script. The macro can receive varadic number of
input, which will automatically generate a bunch of conv + post-op fusion
registeration code. The standard useage of this macro is:
IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(PostOpName, ExtraParametersOfPostOps)

If your post op dose not bring any extra parameter to the schema, you can just
omit the part of the  ExtraParamterofPostOps, for exmaple, when we have Relu as
post-op of convolution, the registeration can write as follow:

IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(relu)

And the above macro will extend to the below expression:

    IPEX_JIT_OP_REGISTER(
      "xpu::q_conv2d_relu(Tensor input,
    __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight,
float output_scale, int output_zpoint) -> Tensor", "q_conv2d_relu",
      at::AtenIpexTypeXPU::q_conv2d_relu),

    IPEX_JIT_OP_REGISTER(
      "xpu::conv2d_relu(Tensor input, Tensor weight, Tensor? bias, int[2]
stride, int[2] padding, int[2] dilation, int groups) -> Tensor",
"convolution_relu", at::AtenIpexTypeXPU::convolution_relu),

    IPEX_JIT_OP_REGISTER(
      "xpu::_convolution_relu(Tensor input, Tensor weight, Tensor? bias, int[]
    stride, int[] padding, int[] dilation, bool transpose, int[] output_padding,
int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool
allow_tf32)
    -> Tensor"
      "_convolution_relu",
      at::AtenIpexTypeXPU::_convolution_relu),

when we have post-ops that will bing extra parameter to the fusion schema, we
can wrap the extra parameter in macro EXTRA_SCHEMA() as second parameter. for
example, the leaky relu will bring "Scalar negative_slope" as extra parameter to
all the convolution fusions, and we are suppose to register it as:

    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(leaky_relu, EXTRA_SCHEMA("Scalar
    negative_slope"))

And the above macro will add the extra parameter at the end of all the
convolution-post-ops's schema, it will extend like:

    IPEX_JIT_OP_REGISTER(
      "xpu::q_conv2d_leaky_relu(Tensor input,
    __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight,
float output_scale, int output_zpoint, Scalar negative_slope) -> Tensor",
      "q_conv2d_leaky_relu",
      at::AtenIpexTypeXPU::q_conv2d_leaky_relu),

    IPEX_JIT_OP_REGISTER(
      "xpu::conv2d_leaky_relu(Tensor input, Tensor weight, Tensor? bias, int[2]
    stride, int[2] padding, int[2] dilation, int groups, Scalar negative_slope)
-> Tensor", "convolution_leaky_relu",
at::AtenIpexTypeXPU::convolution_leaky_relu),

    IPEX_JIT_OP_REGISTER(
      "xpu::_convolution_leaky_relu(Tensor input, Tensor weight, Tensor? bias,
int[] stride, int[] padding, int[] dilation, bool transpose, int[]
output_padding, int groups, bool benchmark, bool deterministic, bool
cudnn_enabled, bool allow_tf32, Scalar negative_slope) -> Tensor"
      "_convolution_leaky_relu",
      at::AtenIpexTypeXPU::_convolution_leaky_relu),


However, there are some restrictions worth to note, the implementation function
of all the registerations, should always
    1. exist in the at::AtenIpexTypeXPU namespace
    2. have exactly same signiture against the extended schema

*/

// NOTE: in some cases, quantized pass can bring different schema change
// compared with non-quantized pass. For example of convolution + sum fusion, we
// have
//
//        IPEX_CONV_JIT_OP_REGISTER(sum,
//           EXTRA_SCHEMA("Tensor(a!) accumu, *, Scalar scale")),

// for non-quantized pass registration and

//        IPEX_QCONV2D_JIT_OP_REGISTER(sum,
//           EXTRA_SCHEMA( "Tensor(a!) accumu, *, Scalar sum_scale, Scalar
//           sum_zpoint")),

// for quantized pass registration.

#define IPEX_CONV_JIT_OP_REGISTER(func, ...)      \
  IPEX_CONV2D_JIT_OP_REGISTER(func, __VA_ARGS__), \
      IPEX__CONV_JIT_OP_REGISTER(func, __VA_ARGS__)

#define IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(func, ...) \
  IPEX_QCONV2D_JIT_OP_REGISTER(func, __VA_ARGS__),     \
      IPEX_CONV_JIT_OP_REGISTER(func, __VA_ARGS__)

} // namespace

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

at::Tensor toOptionalTensor(const IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

using namespace at::native;

/* Implements intrinsic jit op fusion function.

Jit op fusion function will replace the related op inside jit ir graph during
jit pass, and thus enhance the performance of jitscript model. To enable fusion
function replacement, developers need to explicitly declare the fusion rule in
dnnlRules, and specify the fusion components and its result.

example:

1. Declare fusion function's symbol:

In csrc/jit/accelerated_ops.h
static auto pad_conv2d_sym = Symbol::fromQualString("xpu::pad_conv2d");

2. Define the fusion rule in dnnlRules:

In csrc/jit/fusion_pass.cpp
OpFuser::RuleTab OpFuser::dnnlRules = {
...
{{aten::constant_pad_nd, aten::conv2d}, xpu::pad_conv2d_sym},
...
};

3. Declare fusion function's signiture in header file

In csrc/intrinsic/intrinsic.h
at::Tensor pad_convolution(
    const at::Tensor& input,
    at::IntArrayRef pad_nd,
    Scalar value,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

4. Define its fusion function's implementation

In implementation file (this case is csrc/aten/operator/Conv.cpp)
Tensor pad_convolution(
    const Tensor& input_r,
    IntArrayRef pad_nd,
    Scalar value,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    int64_t groups_) {
      ......                (defination)
  }

5. Register the fusion function and bind it with its function schema

In csrc/jit/register_dnnl_jit_ops.cpp
RegisterOperators op({
  ...
  IPEX_JIT_OP_REGISTER(
      "xpu::pad_conv2d(Tensor input, int[] pad_nd, Scalar value, Tensor weight,
Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int
groups=1) -> Tensor", "xpu::pad_conv2d", AtenIpexTypeXPU::pad_convolution),
  ...
});

6. write a UT to verify the functionality of fusion function.
In test_fusion.py/test_fusion_quant.py

def test_pad_conv_fusion(self, dtype=torch.float):
  ...

*/
RegisterOperators op({
    //    Operator(
    //      "dnnl::reorder(Tensor self) -> Tensor",
    //      [](const Node* node) -> Operation {
    //        return [node] (Stack* stack) {
    //          auto* enode = reinterpret_cast<const NodeExt *>(node);
    //          auto from = enode->inputFormat(0);
    //          auto to = enode->inputFormat(1);
    //          auto groups = enode->getGroupInfo();
    //
    //          auto result = torch::jit::xpu::reorder(
    //              (std::move(peek(stack, 0, 1))).toTensor(), from, to,
    //              groups);
    //          drop(stack, 1);
    //          pack(stack, std::move(result));
    //        };
    //      },
    //      aliasAnalysisFromSchema()
    //      ),

    // --------------------------------------------------
    // register convolution related fusion pattern
    // --------------------------------------------------

    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(sigmoid),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(relu),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(sqrt),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(abs),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(tanh),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(square),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(exp),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(log),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(round),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(log_sigmoid),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(hardswish),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(mish),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(silu),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(gelu),
    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(hardsigmoid),

    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(
        leaky_relu,
        EXTRA_SCHEMA("Scalar negative_slope")),

    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(pow, EXTRA_SCHEMA("Scalar exponent")),

    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(
        hardtanh,
        EXTRA_SCHEMA("Scalar minval, Scalar maxval")),

    IPEX_GENERAL_CONV2D_JIT_OP_REGISTER(
        elu,
        EXTRA_SCHEMA("Scalar alpha=1, Scalar scale=1, Scalar input_scale=1")),

    // note: It is found that convolution + sum fusion may result with different
    // schema when it comes with INT8 path.
    IPEX_CONV_JIT_OP_REGISTER(
        sum,
        EXTRA_SCHEMA("Tensor(a!) accumu, *, Scalar scale")),

    IPEX_QCONV2D_JIT_OP_REGISTER(
        sum,
        EXTRA_SCHEMA(
            "Tensor(a!) accumu, *, Scalar sum_scale, Scalar sum_zpoint")),

    IPEX_CONV_JIT_OP_REGISTER(
        sum_relu,
        EXTRA_SCHEMA("Tensor(a!) accumu, *, Scalar scale")),

    IPEX_QCONV2D_JIT_OP_REGISTER(
        sum_relu,
        EXTRA_SCHEMA(
            "Tensor(a!) accumu, *, Scalar sum_scale, Scalar sum_zpoint")),

    // -------------------------------------------------
    // register convolution related fusion pattern
    // -------------------------------------------------

    // -------------------------------------------------
    // register linear related fusion pattern
    // -------------------------------------------------

    IPEX_LINEAR_JIT_OP_REGISTER(sigmoid),
    IPEX_LINEAR_JIT_OP_REGISTER(relu),
    IPEX_LINEAR_JIT_OP_REGISTER(sqrt),
    IPEX_LINEAR_JIT_OP_REGISTER(abs),
    IPEX_LINEAR_JIT_OP_REGISTER(tanh),
    IPEX_LINEAR_JIT_OP_REGISTER(square),
    IPEX_LINEAR_JIT_OP_REGISTER(exp),
    IPEX_LINEAR_JIT_OP_REGISTER(log),
    IPEX_LINEAR_JIT_OP_REGISTER(round),
    IPEX_LINEAR_JIT_OP_REGISTER(log_sigmoid),
    IPEX_LINEAR_JIT_OP_REGISTER(hardswish),
    IPEX_LINEAR_JIT_OP_REGISTER(mish),
    IPEX_LINEAR_JIT_OP_REGISTER(silu),
    // IPEX_LINEAR_JIT_OP_REGISTER(gelu),
    IPEX_LINEAR_JIT_OP_REGISTER(hardsigmoid),

    IPEX_LINEAR_JIT_OP_REGISTER(
        leaky_relu,
        EXTRA_SCHEMA("Scalar negative_slope")),

    IPEX_LINEAR_JIT_OP_REGISTER(pow, EXTRA_SCHEMA("Scalar exponent")),

    IPEX_LINEAR_JIT_OP_REGISTER(
        hardtanh,
        EXTRA_SCHEMA("Scalar minval, Scalar maxval")),

    IPEX_LINEAR_JIT_OP_REGISTER(
        elu,
        EXTRA_SCHEMA("Scalar alpha=1, Scalar scale=1, Scalar input_scale=1")),

    IPEX_LINEAR_JIT_OP_REGISTER(
        sum,
        EXTRA_SCHEMA("Tensor(a!) accumu, *, Scalar alpha")),

    // -------------------------------------------------
    // register linear related fusion pattern
    // -------------------------------------------------

    IPEX_JIT_OP_REGISTER(
        "xpu::conv2d_binary_mul(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups, Tensor binary) -> Tensor",
        "xpu::conv2d_binary_mul",
        at::AtenIpexTypeXPU::convolution_binary_mul),

    IPEX_JIT_OP_REGISTER(
        "xpu::permute_contiguous(Tensor self, int[] dims, MemoryFormat memory_format=contiguous_format) -> Tensor(a)",
        "xpu::permute_contiguous",
        AtenIpexTypeXPU::permute_contiguous),

    IPEX_JIT_OP_REGISTER(
        "xpu::pad_conv2d(Tensor input, int[] pad_nd, Scalar value, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
        "xpu::pad_conv2d",
        AtenIpexTypeXPU::pad_convolution),

    IPEX_JIT_OP_REGISTER(
        "xpu::q_conv2d_dequantize(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float conv_scale, int conv_zpoint) -> Tensor",
        "xpu::q_conv2d",
        AtenIpexTypeXPU::q_conv2d_dequantize),

    IPEX_JIT_OP_REGISTER(
        "xpu::softplus_tanh(Tensor self, Scalar beta, Scalar threshold) -> Tensor",
        "xpu::softplus_tanh",
        AtenIpexTypeXPU::softplus_tanh),

    IPEX_JIT_OP_REGISTER(
        "xpu::softplus_tanh_mul(Tensor self, Scalar beta, Scalar threshold, Tensor mul_input) -> Tensor",
        "xpu::softplus_tanh_mul",
        AtenIpexTypeXPU::softplus_tanh_mul),

    IPEX_JIT_OP_REGISTER(
        "xpu::q_conv2d_dequantize_softplus_tanh_mul(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float conv_scale, int conv_zpoint, Scalar beta, Scalar threshold) -> Tensor",
        "xpu::q_conv2d_dequantize_softplus_tanh_mul",
        AtenIpexTypeXPU::q_conv2d_dequantize_softplus_tanh_mul),

    IPEX_JIT_OP_REGISTER(
        "xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float conv_scale, int conv_zpoint, Scalar beta, Scalar threshold, float q_scale, int q_zpoint, ScalarType dtype) -> Tensor",
        "xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize",
        AtenIpexTypeXPU::q_conv2d_dequantize_softplus_tanh_mul_quantize),

    IPEX_JIT_OP_REGISTER(
        "xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize_add(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float conv_scale, int conv_zpoint, Scalar beta, Scalar threshold, float q_scale, int q_zpoint, ScalarType dtype, Tensor qb, double add_scale, int64_t add_zero_point) -> Tensor",
        "xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize_add",
        AtenIpexTypeXPU::q_conv2d_dequantize_softplus_tanh_mul_quantize_add),

    IPEX_JIT_OP_REGISTER(
        "xpu::mul_add(Tensor self, Tensor other, Tensor accumu, Scalar alpha) -> Tensor",
        "xpu::mul_add",
        AtenIpexTypeXPU::mul_add),

    IPEX_JIT_OP_REGISTER(
        "xpu::matmul_add(Tensor m1, Tensor m2, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
        "xpu::matmul_add",
        AtenIpexTypeXPU::matmul_add),

    IPEX_JIT_OP_REGISTER(
        "xpu::trans_matmul(Tensor m2, int dim1, int dim2, Tensor m1) -> Tensor(a!)",
        "xpu::trans_matmul",
        AtenIpexTypeXPU::trans_matmul),

    IPEX_JIT_OP_REGISTER(
        "xpu::t_matmul(Tensor m2, Tensor m1) -> Tensor(a!)",
        "xpu::t_matmul",
        AtenIpexTypeXPU::t_matmul_add),

    IPEX_JIT_OP_REGISTER(
        "xpu::t_matmul_add(Tensor m2, Tensor m1, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
        "xpu::t_matmul_add",
        AtenIpexTypeXPU::t_matmul_add),

    IPEX_JIT_OP_REGISTER(
        "xpu::t_matmul_add_gelu(Tensor m2, Tensor m1, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
        "xpu::t_matmul_add_gelu",
        AtenIpexTypeXPU::t_matmul_add_gelu),

    IPEX_JIT_OP_REGISTER(
        "xpu::t_matmul_add_add(Tensor m2, Tensor m1, Tensor(a!) accumu1, *, Scalar alpha1, Tensor(a!) accumu2, *, Scalar alpha2) -> Tensor(a!)",
        "xpu::t_matmul_add_add",
        AtenIpexTypeXPU::t_matmul_add_add),

    // FIXME: support not only div scalar but div tensor
    IPEX_JIT_OP_REGISTER(
        "xpu::trans_matmul_div(Tensor m2, int dim1, int dim2, Tensor m1, Scalar oscale) -> Tensor(a!)",
        "xpu::trans_matmul_div",
        AtenIpexTypeXPU::trans_matmul_div),

    IPEX_JIT_OP_REGISTER(
        "xpu::linear_gelu(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor ",
        "xpu::linear_gelu",
        AtenIpexTypeXPU::linear_gelu),

    IPEX_JIT_OP_REGISTER(
        "xpu::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool dummy) -> Tensor",
        "xpu::batch_norm",
        torch::jit::xpu::batch_norm),

    IPEX_JIT_OP_REGISTER(
        "xpu::fold_weight(Tensor weight, Tensor? bn_weight, Tensor? running_var, float eps) -> Tensor",
        "xpu::fold_weight",
        torch::jit::xpu::fold_weight),

    IPEX_JIT_OP_REGISTER(
        "xpu::fold_bias(Tensor weight, Tensor? bias, Tensor? bn_weight, Tensor? bn_bias, Tensor? running_mean, Tensor? running_var, float eps) -> Tensor",
        "xpu::fold_bias",
        torch::jit::xpu::fold_bias),

    IPEX_JIT_OP_REGISTER(
        "xpu::dequant_pixelshuffle(Tensor self, int64_t upscale_factor) -> Tensor",
        "xpu::dequant_pixelshuffle",
        torch::jit::xpu::dequant_pixelshuffle),

    IPEX_JIT_OP_REGISTER(
        "xpu::dequant_pixelshuffle_quant(Tensor self, int64_t upscale_factor, double scale, int64_t zero_pad, ScalarType dtype) -> Tensor",
        "xpu::dequant_pixelshuffle_quant",
        torch::jit::xpu::dequant_pixelshuffle_quant),

});
} // namespace jit
} // namespace torch
