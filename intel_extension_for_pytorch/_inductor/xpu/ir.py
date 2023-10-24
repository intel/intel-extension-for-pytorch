import torch
from typing import List, Optional, Any
from torch._inductor.ir import (
    ExternKernelAlloc,
    MutationLayout,
    Layout,
    FixedLayout,
    FlexibleLayout,
    convert_shape_to_inductor,
    ir_node_to_tensor,
    may_convert_to_optional,
    OptionalString,
    OptionalList,
    OptionalScalar,
    TensorBox,
)
from torch._prims_common import (
    make_channels_last_strides_for,
)
from torch._inductor.virtualized import V


def _prepare_convolution_fusion_create(
    cls,
    x: "TensorBox",
    weight: "TensorBox",
    bias: "TensorBox",
    padding_: List[int],
    stride_: List[int],
    dilation_: List[int],
    groups: int,
    transposed: bool = False,
    output_padding_: List[int] = None,
):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for convolution post-op fusion's create function, including deciding the output
    layout (channels first or channels last), realizing inputs and make them etc. The
    function only supports the CPU device since conv post-op fusion kernel is only
    supported on CPU right now.
    """

    # Port from aten/src/ATen/native/ConvUtils.h: _conv_input_size
    def _conv_input_size(
        output_size, weight_size, padding, output_padding, stride, dilation, groups
    ):
        assert len(output_size) == len(weight_size), "Expect input dim == weight dim"
        dim = len(output_size)
        assert dim > 2, "Expect input dim > 2"

        BATCH_DIM = 0
        WEIGHT_INPUT_CHANNELS_DIM = 1
        input_size = []
        input_size.append(output_size[BATCH_DIM])
        input_size.append(weight_size[WEIGHT_INPUT_CHANNELS_DIM] * groups)
        for d in range(2, dim):
            kernel = (weight_size[d] - 1) * dilation[d - 2] + 1
            input_size_d = (
                (output_size[d] - 1) * stride[d - 2]
                - (padding[d - 2] * 2)
                + kernel
                + output_padding[d - 2]
            )
            input_size.append(input_size_d)
        return list(map(int, input_size))

    # The size of prepacked_weight is the prepacked weight size of deconv:
    #   Groups > 1:  [g*o, i/g, ...]
    #   Groups == 1: [o, i, ...]
    # Returns original weight size in [i, o, ...]
    def _original_deconv_weight_size(
        prepacked_weight,
        groups,
    ):
        prepacked_weight_size = prepacked_weight.size()
        dim = len(prepacked_weight_size)
        assert dim > 2, "Expect weight dim > 2"
        if groups > 1:
            weight_size = []
            weight_size.append(prepacked_weight_size[1] * groups)
            weight_size.append(prepacked_weight_size[0] / groups)
            for d in range(2, dim):
                weight_size.append(prepacked_weight_size[d])
        else:
            weight_size = prepacked_weight.transpose(0, 1).size()
        return weight_size

    stride = tuple(stride_)
    padding = tuple(padding_)
    dilation = tuple(dilation_)
    assert isinstance(groups, int)
    output_padding = tuple(output_padding_) if output_padding_ else (0, 0)
    with V.graph.fake_mode:
        x_fake = ir_node_to_tensor(x, guard_shape=True)
        weight_fake = ir_node_to_tensor(weight, guard_shape=True)
        if transposed:
            # When transposed, the size of the prepacked oneDNN weight is different
            # from the PyTorch weight. We're not able to run aten conv with such
            # size. We infer the output size from the input params here:
            weight_size = _original_deconv_weight_size(weight_fake, groups)
            input_size = x_fake.size()
            output_size = _conv_input_size(
                input_size,
                weight_size,
                padding,
                output_padding,
                stride,
                dilation,
                groups,
            )
        else:
            bias_fake = (
                ir_node_to_tensor(bias, guard_shape=True) if bias is not None else bias
            )
            output = torch.ops.aten.convolution(
                x_fake,
                weight_fake,
                bias_fake,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            )
            output_size = output.size()

        req_stride_order = [0] + list(reversed(range(1, len(stride) + 1)))
        req_stride_order = [len(req_stride_order)] + req_stride_order
        output_stride = make_channels_last_strides_for(output_size)

    x = cls.require_stride_order(x, req_stride_order)
    # assert x.get_device().type == "cpu" and weight.get_device().type == "cpu"
    inputs = [x, weight]

    kernel_layout = FixedLayout(
        x.get_device(),
        x.get_dtype(),
        convert_shape_to_inductor(output_size),
        convert_shape_to_inductor(output_stride),
    )
    constant_args = [padding, stride, dilation, groups]
    if transposed:
        constant_args.insert(1, output_padding)

    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return inputs, constant_args, kernel_layout, req_stride_order


class ConvolutionUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kernel="torch.ops.torch_ipex._convolution_pointwise",
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            kernel="torch.ops.torch_ipex._convolution_pointwise",
            cpp_kernel="torch_ipex::_convolution_pointwise",
        )
        self.cpp_kernel_key = "convolution_pointwise"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.kernel,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        attr,
        scalars,
        algorithm,
    ):
        (inputs, constant_args, kernel_layout, _) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        optional_string = OptionalString()
        optional_list = OptionalList()
        constant_args = constant_args + [
            attr,
            may_convert_to_optional(optional_list, scalars),
            may_convert_to_optional(optional_string, algorithm),
        ]
        return ConvolutionUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


class LinearUnary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            kernel="torch.ops.torch_ipex._linear_pointwise",
            cpp_kernel="torch_ipex::_linear_pointwise",
        )
        self.cpp_kernel_key = "linear_pointwise"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.kernel,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
        )

    @classmethod
    def create(cls, x, w, b, attr, scalars, algorithm):
        x = cls.require_stride1(cls.realize_input(x))
        w = cls.require_stride1(cls.realize_input(w))

        *m, ic = x.get_size()
        # After freezing pass, w should be [ic, oc] shape, format is ba
        ic, oc = w.get_size()

        inputs = [x, w]
        constant_args = [attr, scalars if scalars else [-1], algorithm]
        if b is not None:
            b = cls.require_stride1(cls.realize_input(b))
            inputs.append(b)
        else:
            constant_args.insert(0, None)

        return LinearUnary(
            layout=FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(m) + [oc],
            ),
            inputs=inputs,
            constant_args=constant_args,
        )

    def apply_constraint(self):
        pass


class LinearBinary(ExternKernelAlloc):
    kernel = "torch.ops.torch_ipex._linear_pointwise.binary"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            kernel="torch.ops.torch_ipex._linear_pointwise.binary",
            cpp_kernel="torch_ipex::_linear_pointwise",
        )
        self.cpp_kernel_overlad_name = "binary"
        self.cpp_kernel_key = "linear_pointwise_binary"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& other_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                c10::string_view attr)
        """

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.kernel,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overlad_name,
        )

    @classmethod
    def create(cls, x, y, w, b, attr):
        x = cls.require_stride1(cls.realize_input(x))
        y = cls.require_stride1(cls.realize_input(y))
        w = cls.require_stride1(cls.realize_input(w))

        *m, ic = x.get_size()
        # After freezing pass, w should be [ic, oc] shape, format is ba
        ic, oc = w.get_size()

        inputs = [x, y, w]
        constant_args = [attr]
        if b is not None:
            b = cls.require_stride1(cls.realize_input(b))
            inputs.append(b)
        else:
            constant_args.insert(0, b)

        return LinearBinary(
            layout=FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(m) + [oc],
            ),
            inputs=inputs,
            constant_args=constant_args,
        )

    def apply_constraint(self):
        pass


class ConvolutionBinary(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        cpp_constant_args=(),
    ):
        super().__init__(
            layout,
            inputs,
            constant_args,
            None,
            kernel="torch.ops.torch_ipex._convolution_pointwise.binary",
            cpp_kernel="torch_ipex::_convolution_pointwise",
        )
        self.cpp_kernel_overlad_name = "binary"
        self.cpp_kernel_key = "convolution_pointwise_binary"
        self.cpp_op_schema = """
            at::Tensor(
                const at::Tensor& input_t,
                at::Tensor& other_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> unary_attr,
                torch::List<c10::optional<at::Scalar>> unary_scalars,
                c10::optional<c10::string_view> unary_algorithm)"""
        self.cpp_constant_args = cpp_constant_args

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.kernel,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overlad_name,
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List[Any]],
        unary_algorithm: Optional[str],
    ):
        (
            inputs,
            constant_args,
            kernel_layout,
            req_stride_order,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        optional_scalar = OptionalScalar()
        optional_string = OptionalString()
        optional_list = OptionalList()
        constant_args = constant_args + [
            binary_attr,
            may_convert_to_optional(optional_scalar, binary_alpha),
            may_convert_to_optional(optional_string, unary_attr),
            may_convert_to_optional(optional_list, unary_scalars),
            may_convert_to_optional(optional_string, unary_algorithm),
        ]
        return ConvolutionBinary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
        )


class ConvolutionBinaryInplace(ExternKernelAlloc):
    def __init__(
        self,
        kernel_layout,
        inputs,
        constant_args=(),
    ):
        # Due to constrain of op.call, other (Tensor&) should be at input[0]
        reordered_inputs = [inputs[1], inputs[0]] + inputs[2:]

        super().__init__(
            kernel_layout,
            reordered_inputs,
            constant_args,
            None,
            kernel="torch.ops.torch_ipex._convolution_pointwise_.binary",
            cpp_kernel="torch_ipex::_convolution_pointwise_",
        )
        self.cpp_kernel_overlad_name = "binary"
        self.cpp_kernel_key = "convolution_pointwise_binary_"
        # TODO: op.call: input[0] should be at::Tensor&
        self.cpp_op_schema = """
            at::Tensor&(
                at::Tensor& other_t,
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view binary_attr,
                c10::optional<at::Scalar> alpha,
                c10::optional<c10::string_view> unary_attr,
                torch::List<c10::optional<at::Scalar>> unary_scalars,
                c10::optional<c10::string_view> unary_algorithm)"""

    def codegen(self, wrapper):
        wrapper.generate_extern_kernel_alloc_and_find_schema_if_needed(
            self.get_name(),
            self.kernel,
            self.codegen_args(),
            self.cpp_op_schema,
            self.cpp_kernel_key,
            self.cpp_kernel_overlad_name,
        )

    def get_mutation_names(self):
        assert isinstance(self.layout, MutationLayout)
        return (self.layout.target.get_name(),)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List[Any]],
        unary_algorithm: Optional[str],
    ):
        (
            inputs,
            constant_args,
            _,
            req_stride_order,
        ) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        other = cls.require_stride_order(other, req_stride_order)
        inputs.insert(1, other)
        optional_scalar = OptionalScalar()
        optional_string = OptionalString()
        optional_list = OptionalList()
        constant_args = constant_args + [
            binary_attr,
            may_convert_to_optional(optional_scalar, binary_alpha),
            may_convert_to_optional(optional_string, unary_attr),
            may_convert_to_optional(optional_list, unary_scalars),
            may_convert_to_optional(optional_string, unary_algorithm),
        ]
        return ConvolutionBinaryInplace(
            kernel_layout=MutationLayout(inputs[1]),
            inputs=inputs,
            constant_args=constant_args,
        )
