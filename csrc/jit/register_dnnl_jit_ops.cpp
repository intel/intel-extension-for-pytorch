#include <quantized/QUtil.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <vector>
#include "ATen/core/ivalue.h"
#include "ATen/native/quantized/cpu/conv_packed_params.h"
#include "accelerated_ops.h"
#include "c10/core/Scalar.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "dpcpp_ops.h"

namespace {
// using namespace torch;
using namespace at::native;
using namespace torch::jit;
struct TensorLValue;
struct TensorNeedPlain;
template <typename T>
struct IvalueConvert {
  T operator()(c10::IValue&) {}
};

#define DEFINE_CONVERTER(T, mf)           \
  template <>                             \
  struct IvalueConvert<T> {               \
    T operator()(c10::IValue& i_value) {  \
      return i_value.mf;                  \
    }                                     \
    T operator()(c10::IValue&& i_value) { \
      return i_value.mf;                  \
    }                                     \
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

DEFINE_CONVERTER(at::Tensor, toTensor())
DEFINE_CONVERTER(std::vector<int64_t>, toIntVector())
DEFINE_CONVERTER(at::Scalar, toScalar())
DEFINE_CONVERTER(at::ScalarType, toScalarType())
DEFINE_CONVERTER(at::MemoryFormat, toMemoryFormat())
DEFINE_CONVERTER(at::QScheme, toQScheme())
DEFINE_CONVERTER(c10::Device, toDevice())
DEFINE_CONVERTER(c10::intrusive_ptr<ivalue::Object>, toObject())
DEFINE_CONVERTER(c10::Stream, toStream())
DEFINE_CONVERTER(c10::Layout, toLayout())
DEFINE_CONVERTER(int, toInt())
DEFINE_CONVERTER(float, toDouble())
DEFINE_CONVERTER(double, toDouble())
DEFINE_CONVERTER(bool, toBool())
DEFINE_CONVERTER(c10::complex<double>, toComplexDouble())
DEFINE_CONVERTER(at::Dimname, toDimname())
DEFINE_CUSTOMTYPE_CONVERT(ConvPackedParamsBase<2>)

template <>
struct IvalueConvert<c10::optional<at::Tensor>> {
  at::Tensor operator()(IValue& i_value) {
    return i_value.isNone() ? at::Tensor() : i_value.toTensor();
  }
  at::Tensor operator()(IValue&& i_value) {
    return i_value.isNone() ? at::Tensor() : i_value.toTensor();
  }
};

template <>
struct IvalueConvert<TensorLValue> {
  at::Tensor& operator()(c10::IValue& i_value) {
    val_ = i_value.toTensor();
    return val_;
  }
  at::Tensor& operator()(c10::IValue&& i_value) {
    val_ = i_value.toTensor();
    return val_;
  }

  at::Tensor val_;
};

template <typename... Args>
constexpr int typeCounter() {
  return sizeof...(Args);
}

template <int n, typename Func, typename... Args, size_t... N>
torch::jit::Operation generateJitOperation(
    Func func,
    std::index_sequence<N...> seq) {
  return [=](Stack& stack) {
    auto result = func(IvalueConvert<Args>()(std::move(peek(stack, N, n)))...);
    drop(stack, n);
    pack(stack, std::move(result));
  };
}

#define IPEX_JIT_OP_REGISTER(schema, func, ...)                      \
  Operator(                                                          \
      schema,                                                        \
      [](const Node* node) -> Operation {                            \
        constexpr int n = typeCounter<__VA_ARGS__>();                \
        return generateJitOperation<n, decltype(func), __VA_ARGS__>( \
            func, std::make_index_sequence<n>());                    \
      },                                                             \
      aliasAnalysisFromSchema())

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

RegisterOperators op(
    {
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
        IPEX_JIT_OP_REGISTER(
            "xpu::pad_conv2d(Tensor input, int[] pad_nd, Scalar value, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
            torch::jit::xpu::pad_conv2d,
            Tensor,
            std::vector<int64_t>,
            Scalar,
            Tensor,
            c10::optional<Tensor>,
            std::vector<int64_t>,
            std::vector<int64_t>,
            std::vector<int64_t>,
            int),
        IPEX_JIT_OP_REGISTER(
            "xpu::conv2d_relu(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
            torch::jit::xpu::conv2d_relu,
            Tensor,
            Tensor,
            c10::optional<Tensor>,
            std::vector<int64_t>,
            std::vector<int64_t>,
            std::vector<int64_t>,
            int),
        IPEX_JIT_OP_REGISTER(
            "xpu::conv2d_sigmoid(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
            torch::jit::xpu::conv2d_sigmoid,
            Tensor,
            Tensor,
            c10::optional<Tensor>,
            std::vector<int64_t>,
            std::vector<int64_t>,
            std::vector<int64_t>,
            int),
        Operator(
            "xpu::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool dummy) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::batch_norm(
                    (std::move(peek(stack, 0, 9))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 1, 9))),
                    toOptionalTensor(std::move(peek(stack, 2, 9))),
                    toOptionalTensor(std::move(peek(stack, 3, 9))),
                    toOptionalTensor(std::move(peek(stack, 4, 9))),
                    (std::move(peek(stack, 5, 9))).toBool(),
                    (std::move(peek(stack, 6, 9))).toDouble(),
                    (std::move(peek(stack, 7, 9))).toDouble(),
                    (std::move(peek(stack, 8, 9))).toBool());
                drop(stack, 9);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::fold_weight(Tensor weight, Tensor? bn_weight, Tensor? running_var, float eps) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::fold_weight(
                    (std::move(peek(stack, 0, 4))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 1, 4))),
                    toOptionalTensor(std::move(peek(stack, 2, 4))),
                    (std::move(peek(stack, 3, 4))).toDouble());
                drop(stack, 4);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::fold_bias(Tensor weight, Tensor? bias, Tensor? bn_weight, Tensor? bn_bias, Tensor? running_mean, Tensor? running_var, float eps) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::fold_bias(
                    (std::move(peek(stack, 0, 7))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 1, 7))),
                    toOptionalTensor(std::move(peek(stack, 2, 7))),
                    toOptionalTensor(std::move(peek(stack, 3, 7))),
                    toOptionalTensor(std::move(peek(stack, 4, 7))),
                    toOptionalTensor(std::move(peek(stack, 5, 7))),
                    (std::move(peek(stack, 6, 7))).toDouble());
                drop(stack, 7);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::conv2d_sum(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto output = (std::move(peek(stack, 7, 9))).toTensor();
                auto result = torch::jit::xpu::conv2d_sum(
                    output,
                    (std::move(peek(stack, 0, 9))).toTensor(),
                    (std::move(peek(stack, 1, 9))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 2, 9))),
                    (std::move(peek(stack, 3, 9))).toIntVector(),
                    (std::move(peek(stack, 4, 9))).toIntVector(),
                    (std::move(peek(stack, 5, 9))).toIntVector(),
                    (std::move(peek(stack, 6, 9))).toInt(),
                    (std::move(peek(stack, 8, 9))).toScalar());
                drop(stack, 9);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::conv2d_sum_relu(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto output = (std::move(peek(stack, 7, 9))).toTensor();
                auto result = torch::jit::xpu::conv2d_sum_relu(
                    output,
                    (std::move(peek(stack, 0, 9))).toTensor(),
                    (std::move(peek(stack, 1, 9))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 2, 9))),
                    (std::move(peek(stack, 3, 9))).toIntVector(),
                    (std::move(peek(stack, 4, 9))).toIntVector(),
                    (std::move(peek(stack, 5, 9))).toIntVector(),
                    (std::move(peek(stack, 6, 9))).toInt(),
                    (std::move(peek(stack, 8, 9))).toScalar());
                drop(stack, 9);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::matmul_add(Tensor m1, Tensor m2, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto accumu1 = (std::move(peek(stack, 2, 4))).toTensor();
                at::Tensor accumu2;
                auto result = torch::jit::xpu::matmul_fusion_variants(
                    accumu1,
                    accumu2,
                    (std::move(peek(stack, 0, 4))).toTensor(),
                    (std::move(peek(stack, 1, 4))).toTensor(),
                    1.f,
                    (std::move(peek(stack, 3, 4))).toScalar().to<float>(),
                    0.f,
                    true);
                drop(stack, 4);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::trans_matmul(Tensor m2, int dim1, int dim2, Tensor m1) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                at::Tensor accumu1;
                at::Tensor accumu2;
                auto result = torch::jit::xpu::matmul_fusion_variants(
                    accumu1,
                    accumu2,
                    (std::move(peek(stack, 3, 4))).toTensor(),
                    (std::move(peek(stack, 0, 4))).toTensor(),
                    1.f,
                    0.f,
                    0.f,
                    false);
                drop(stack, 4);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::t_matmul(Tensor m2, Tensor m1) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                at::Tensor accumu1;
                at::Tensor accumu2;
                auto result = torch::jit::xpu::matmul_fusion_variants(
                    accumu1,
                    accumu2,
                    (std::move(peek(stack, 1, 2))).toTensor(),
                    (std::move(peek(stack, 0, 2))).toTensor(),
                    1.f,
                    0.f,
                    0.f,
                    false);
                drop(stack, 2);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::t_matmul_add_dropout(Tensor m2, Tensor m1, Tensor(a!) accumu, *, Scalar alpha, double p, bool train) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto accumu1 = (std::move(peek(stack, 4, 8))).toTensor();
                at::Tensor accumu2;
                auto result = torch::jit::xpu::matmul_fusion_variants_dropout(
                    accumu1,
                    accumu2,
                    (std::move(peek(stack, 3, 8))).toTensor(),
                    (std::move(peek(stack, 0, 8))).toTensor(),
                    1.f,
                    (std::move(peek(stack, 5, 8))).toScalar().to<float>(),
                    0.f,
                    false,
                    (std::move(peek(stack, 6, 8))).toDouble(),
                    (std::move(peek(stack, 7, 8))).toBool(),
                    true);
                drop(stack, 8);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::t_matmul_add(Tensor m2, Tensor m1, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto accumu1 = (std::move(peek(stack, 2, 4))).toTensor();
                at::Tensor accumu2;
                auto result = torch::jit::xpu::matmul_fusion_variants(
                    accumu1,
                    accumu2,
                    (std::move(peek(stack, 1, 4))).toTensor(),
                    (std::move(peek(stack, 0, 4))).toTensor(),
                    1.f,
                    (std::move(peek(stack, 3, 4))).toScalar().to<float>(),
                    0.f,
                    false);
                drop(stack, 4);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::t_matmul_add_gelu(Tensor m2, Tensor m1, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto accumu1 = (std::move(peek(stack, 2, 4))).toTensor();
                at::Tensor accumu2;
                auto result = torch::jit::xpu::matmul_fusion_variants_gelu(
                    accumu1,
                    accumu2,
                    (std::move(peek(stack, 1, 4))).toTensor(),
                    (std::move(peek(stack, 0, 4))).toTensor(),
                    1.f,
                    (std::move(peek(stack, 3, 4))).toScalar().to<float>(),
                    0.f,
                    false);
                drop(stack, 4);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::t_matmul_add_add(Tensor m2, Tensor m1, Tensor(a!) accumu1, *, Scalar alpha1, Tensor(a!) accumu2, *, Scalar alpha2) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto accumu1 = (std::move(peek(stack, 2, 6))).toTensor();
                auto accumu2 = (std::move(peek(stack, 4, 6))).toTensor();
                auto result = torch::jit::xpu::matmul_fusion_variants(
                    accumu1,
                    accumu2,
                    (std::move(peek(stack, 1, 6))).toTensor(),
                    (std::move(peek(stack, 0, 6))).toTensor(),
                    1.0f,
                    (std::move(peek(stack, 3, 6))).toScalar().to<float>(),
                    (std::move(peek(stack, 5, 6))).toScalar().to<float>(),
                    false);
                drop(stack, 6);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        // FIXME: support not only div scalar but div tensor
        Operator(
            "xpu::trans_matmul_div(Tensor m2, int dim1, int dim2, Tensor m1, Scalar oscale) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                at::Tensor accumu1;
                at::Tensor accumu2;
                auto result = torch::jit::xpu::matmul_fusion_variants(
                    accumu1,
                    accumu2,
                    (std::move(peek(stack, 3, 5))).toTensor(),
                    (std::move(peek(stack, 0, 5))).toTensor(),
                    1.f / (std::move(peek(stack, 4, 5))).toScalar().to<float>(),
                    0.f,
                    0.f,
                    false);
                drop(stack, 5);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::trans_matmul_scale_add(Tensor m2, int dim1, int dim2, Tensor m1, Scalar oscale, Tensor accumu, Scalar alpha) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto accumu1 = (std::move(peek(stack, 5, 7))).toTensor();
                at::Tensor accumu2;
                auto result = torch::jit::xpu::matmul_fusion_variants(
                    accumu1,
                    accumu2,
                    (std::move(peek(stack, 3, 7))).toTensor(),
                    (std::move(peek(stack, 0, 7))).toTensor(),
                    1.f / (std::move(peek(stack, 4, 7))).toScalar().to<float>(),
                    (std::move(peek(stack, 6, 7))).toScalar().to<float>(),
                    0.f,
                    false);
                drop(stack, 7);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::mul_add(Tensor self, Tensor other, Tensor accumu, Scalar alpha) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::mul_add(
                    (std::move(peek(stack, 0, 4))).toTensor(),
                    (std::move(peek(stack, 1, 4))).toTensor(),
                    (std::move(peek(stack, 2, 4))).toTensor(),
                    (std::move(peek(stack, 3, 4))).toScalar());
                drop(stack, 4);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::dequant_pixelshuffle(Tensor self, int64_t upscale_factor) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::dequant_pixelshuffle(
                    (std::move(peek(stack, 0, 2))).toTensor(),
                    (std::move(peek(stack, 1, 2))).toInt());
                return 0;
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::dequant_pixelshuffle_quant(Tensor self, int64_t upscale_factor, double scale, int64_t zero_pad, ScalarType dtype) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::dequant_pixelshuffle_quant(
                    (std::move(peek(stack, 0, 5))).toTensor(),
                    (std::move(peek(stack, 1, 5))).toInt(),
                    (std::move(peek(stack, 2, 5))).toDouble(),
                    (std::move(peek(stack, 3, 5))).toInt(),
                    (std::move(peek(stack, 4, 5))).toScalarType());
                drop(stack, 5);
                pack(stack, std::move(result));
                return 0;
              };
            },
            aliasAnalysisFromSchema()),
        IPEX_JIT_OP_REGISTER(
            "xpu::q_conv2d_sum_relu(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float conv_scale, int conv_zpoint, Tensor(a!) accumu, *, float sum_scale, int sum_zpoint) -> Tensor(a!)",
            torch::jit::xpu::q_conv2d_sum_relu,
            Tensor,
            ConvPackedParamsBase<2>,
            double,
            int,
            TensorLValue,
            double,
            int),
        IPEX_JIT_OP_REGISTER(
            "xpu::q_conv2d_leaky_relu(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zpoint, Scalar negative_slope) -> Tensor(a!)",
            torch::jit::xpu::q_conv2d_leaky_relu,
            Tensor,
            ConvPackedParamsBase<2>,
            double,
            int,
            Scalar),
        IPEX_JIT_OP_REGISTER(
            "xpu::q_conv2d_sigmoid(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zpoint) -> Tensor(a!)",
            torch::jit::xpu::q_conv2d_sigmoid,
            Tensor,
            ConvPackedParamsBase<2>,
            double,
            int),
        Operator(
            "xpu::q_conv2d_dequantize(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float conv_scale, int conv_zpoint) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::q_conv2d_dequantize(
                    (std::move(peek(stack, 0, 4))).toTensor(),
                    (std::move(peek(stack, 1, 4)))
                        .toCustomClass<ConvPackedParamsBase<2>>(),
                    (std::move(peek(stack, 2, 4))).toDouble(),
                    (std::move(peek(stack, 3, 4))).toInt());
                drop(stack, 4);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::softplus_tanh(Tensor self, Scalar beta, Scalar threshold) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::softplus_tanh(
                    (std::move(peek(stack, 0, 3))).toTensor(),
                    (std::move(peek(stack, 1, 3))).toScalar(),
                    (std::move(peek(stack, 2, 3))).toScalar());
                drop(stack, 3);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::softplus_tanh_mul(Tensor self, Scalar beta, Scalar threshold, Tensor mul_input) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::softplus_tanh_mul(
                    (std::move(peek(stack, 0, 4))).toTensor(),
                    (std::move(peek(stack, 1, 4))).toScalar(),
                    (std::move(peek(stack, 2, 4))).toScalar(),
                    (std::move(peek(stack, 3, 4))).toTensor());
                drop(stack, 4);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::q_conv2d_dequantize_softplus_tanh_mul(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float conv_scale, int conv_zpoint, Scalar beta, Scalar threshold) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result =
                    torch::jit::xpu::q_conv2d_dequantize_softplus_tanh_mul(
                        (std::move(peek(stack, 0, 6))).toTensor(),
                        (std::move(peek(stack, 1, 6)))
                            .toCustomClass<ConvPackedParamsBase<2>>(),
                        (std::move(peek(stack, 2, 6))).toDouble(),
                        (std::move(peek(stack, 3, 6))).toInt(),
                        (std::move(peek(stack, 4, 6))).toScalar(),
                        (std::move(peek(stack, 5, 6))).toScalar());
                drop(stack, 6);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        IPEX_JIT_OP_REGISTER(
            "xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float conv_scale, int conv_zpoint, Scalar beta, Scalar threshold, float q_scale, int q_zpoint, ScalarType dtype) -> Tensor",
            torch::jit::xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize,
            Tensor,
            ConvPackedParamsBase<2>,
            double,
            int,
            Scalar,
            Scalar,
            double,
            int,
            ScalarType),
        IPEX_JIT_OP_REGISTER(
            "xpu::permute_contiguous(Tensor self, int[] dims, MemoryFormat memory_format=contiguous_format) -> Tensor(a)",
            torch::jit::xpu::permute_contiguous,
            Tensor,
            std::vector<int64_t>,
            MemoryFormat),
        IPEX_JIT_OP_REGISTER(
            "xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize_add(Tensor input, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float conv_scale, int conv_zpoint, Scalar beta, Scalar threshold, float q_scale, int q_zpoint, ScalarType dtype, Tensor qb, double add_scale, int64_t add_zero_point) -> Tensor",
            torch::jit::xpu::q_conv2d_dequantize_softplus_tanh_mul_quantize_add,
            Tensor,
            ConvPackedParamsBase<2>,
            double,
            int,
            Scalar,
            Scalar,
            double,
            int,
            ScalarType,
            Tensor,
            double,
            int),

        Operator(
            "xpu::t_addmm(Tensor weight, Tensor bias, Tensor input, Scalar beta, Scalar alpha) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::trans_addmm(
                    (std::move(peek(stack, 0, 5))).toTensor(),
                    (std::move(peek(stack, 1, 5))).toTensor(),
                    (std::move(peek(stack, 2, 5))).toTensor(),
                    (std::move(peek(stack, 3, 5))).toScalar(),
                    (std::move(peek(stack, 4, 5))).toScalar());
                drop(stack, 5);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::t_addmm_dropout(Tensor weight, Tensor bias, Tensor input, Scalar beta, Scalar alpha, double p, bool train) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::trans_addmm_dropout(
                    (std::move(peek(stack, 0, 7))).toTensor(),
                    (std::move(peek(stack, 1, 7))).toTensor(),
                    (std::move(peek(stack, 2, 7))).toTensor(),
                    (std::move(peek(stack, 3, 7))).toScalar(),
                    (std::move(peek(stack, 4, 7))).toScalar(),
                    (std::move(peek(stack, 5, 7))).toDouble(),
                    (std::move(peek(stack, 6, 7))).toBool(),
                    true);
                drop(stack, 7);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::linear_gelu(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::linear_gelu(
                    (std::move(peek(stack, 0, 3))).toTensor(),
                    (std::move(peek(stack, 1, 3))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 2, 3))));
                drop(stack, 3);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::t_addmm_relu(Tensor weight, Tensor bias, Tensor input, Scalar beta, Scalar alpha) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::trans_addmm_relu(
                    (std::move(peek(stack, 0, 5))).toTensor(),
                    (std::move(peek(stack, 1, 5))).toTensor(),
                    (std::move(peek(stack, 2, 5))).toTensor(),
                    (std::move(peek(stack, 3, 5))).toScalar(),
                    (std::move(peek(stack, 4, 5))).toScalar());
                drop(stack, 5);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::t_addmm_sigmoid(Tensor weight, Tensor bias, Tensor input, Scalar beta, Scalar alpha) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::trans_addmm_sigmoid(
                    (std::move(peek(stack, 0, 5))).toTensor(),
                    (std::move(peek(stack, 1, 5))).toTensor(),
                    (std::move(peek(stack, 2, 5))).toTensor(),
                    (std::move(peek(stack, 3, 5))).toScalar(),
                    (std::move(peek(stack, 4, 5))).toScalar());
                drop(stack, 5);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::_conv2d_relu(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                auto result = torch::jit::xpu::conv2d_relu(
                    (std::move(peek(stack, 0, 7))).toTensor(),
                    (std::move(peek(stack, 1, 7))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 2, 7))),
                    (std::move(peek(stack, 3, 7))).toIntVector(),
                    (std::move(peek(stack, 4, 7))).toIntVector(),
                    (std::move(peek(stack, 5, 7))).toIntVector(),
                    (std::move(peek(stack, 6, 7))).toInt());
                drop(stack, 7);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::_convolution_relu(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                at::Tensor input = std::move(peek(stack, 0, 13)).toTensor();
                auto result = torch::jit::xpu::_convolution_relu(
                    input,
                    (std::move(peek(stack, 1, 13))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 2, 13))),
                    (std::move(peek(stack, 3, 13))).toIntVector(),
                    (std::move(peek(stack, 4, 13))).toIntVector(),
                    (std::move(peek(stack, 5, 13))).toIntVector(),
                    (std::move(peek(stack, 6, 13))).toBool(),
                    (std::move(peek(stack, 7, 13))).toIntVector(),
                    (std::move(peek(stack, 8, 13))).toInt(),
                    (std::move(peek(stack, 9, 13))).toBool(),
                    (std::move(peek(stack, 10, 13))).toBool(),
                    (std::move(peek(stack, 11, 13))).toBool(),
                    (std::move(peek(stack, 12, 13))).toBool());
                drop(stack, 13);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::_convolution_sum(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transpose, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                at::Tensor input = std::move(peek(stack, 0, 15)).toTensor();
                at::Tensor cum = std::move(peek(stack, 13, 15)).toTensor();
                auto result = torch::jit::xpu::_convolution_sum(
                    input,
                    (std::move(peek(stack, 1, 15))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 2, 15))),
                    (std::move(peek(stack, 3, 15))).toIntVector(),
                    (std::move(peek(stack, 4, 15))).toIntVector(),
                    (std::move(peek(stack, 5, 15))).toIntVector(),
                    (std::move(peek(stack, 6, 15))).toBool(),
                    (std::move(peek(stack, 7, 15))).toIntVector(),
                    (std::move(peek(stack, 8, 15))).toInt(),
                    (std::move(peek(stack, 9, 15))).toBool(),
                    (std::move(peek(stack, 10, 15))).toBool(),
                    (std::move(peek(stack, 11, 15))).toBool(),
                    (std::move(peek(stack, 12, 15))).toBool(),
                    cum,
                    (std::move(peek(stack, 14, 15))).toScalar());
                drop(stack, 15);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::_convolution_sum_relu(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transpose, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, Tensor(a!) accumu, *, Scalar alpha) -> Tensor(a!)",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                at::Tensor input = std::move(peek(stack, 0, 15)).toTensor();
                at::Tensor cum = std::move(peek(stack, 13, 15)).toTensor();
                auto result = torch::jit::xpu::_convolution_sum_relu(
                    input,
                    (std::move(peek(stack, 1, 15))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 2, 15))),
                    (std::move(peek(stack, 3, 15))).toIntVector(),
                    (std::move(peek(stack, 4, 15))).toIntVector(),
                    (std::move(peek(stack, 5, 15))).toIntVector(),
                    (std::move(peek(stack, 6, 15))).toBool(),
                    (std::move(peek(stack, 7, 15))).toIntVector(),
                    (std::move(peek(stack, 8, 15))).toInt(),
                    (std::move(peek(stack, 9, 15))).toBool(),
                    (std::move(peek(stack, 10, 15))).toBool(),
                    (std::move(peek(stack, 11, 15))).toBool(),
                    (std::move(peek(stack, 12, 15))).toBool(),
                    cum,
                    (std::move(peek(stack, 14, 15))).toScalar());
                drop(stack, 15);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),
        Operator(
            "xpu::_convolution_silu(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
            [](const Node* node) -> Operation {
              return [](Stack& stack) {
                at::Tensor input = std::move(peek(stack, 0, 13)).toTensor();
                auto result = torch::jit::xpu::_convolution_silu(
                    input,
                    (std::move(peek(stack, 1, 13))).toTensor(),
                    toOptionalTensor(std::move(peek(stack, 2, 13))),
                    (std::move(peek(stack, 3, 13))).toIntVector(),
                    (std::move(peek(stack, 4, 13))).toIntVector(),
                    (std::move(peek(stack, 5, 13))).toIntVector(),
                    (std::move(peek(stack, 6, 13))).toBool(),
                    (std::move(peek(stack, 7, 13))).toIntVector(),
                    (std::move(peek(stack, 8, 13))).toInt(),
                    (std::move(peek(stack, 9, 13))).toBool(),
                    (std::move(peek(stack, 10, 13))).toBool(),
                    (std::move(peek(stack, 11, 13))).toBool(),
                    (std::move(peek(stack, 12, 13))).toBool());
                drop(stack, 13);
                pack(stack, std::move(result));
              };
            },
            aliasAnalysisFromSchema()),

    });
} // namespace jit
} // namespace torch
