import torch
import torch.nn as nn
from typing import Optional
from intel_extension_for_pytorch.nn.utils._weight_prepack import (
    _IPEXLinear,
)
from .utils import IPEXRuntimeCustomOps, IPEXCustomOpType


class IPEXLinearFusion(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.linear_fusion = None
        self.device_type = None
        self.runtime_ops = IPEXRuntimeCustomOps()

    def init_on_device(self, x, op_type):
        self.device_type = x.device.type
        self.linear_fusion = self.runtime_ops.get_module_from_device(
            self.device_type, op_type, False
        )(
            self.linear,
            tpp=(
                self.linear.use_tpp if isinstance(self.linear, _IPEXLinear) else False
            ),
        )


class IPEXLinear2Fusion(nn.Module):
    def __init__(self, linear_1, linear_2):
        super().__init__()
        self.linear_1 = linear_1
        self.linear_2 = linear_2
        self.linear_fusion = None
        self.device_type = None
        self.runtime_ops = IPEXRuntimeCustomOps()

    def init_on_device(self, x, op_type):
        self.device_type = x.device.type
        self.linear_fusion = self.runtime_ops.get_module_from_device(
            self.device_type, op_type, False
        )(
            self.linear_1,
            self.linear_2,
            tpp=(
                self.linear_1.use_tpp and self.linear_2.use_tpp
                if isinstance(self.linear_1, _IPEXLinear)
                and isinstance(self.linear_2, _IPEXLinear)
                else False
            ),
        )


class LinearSilu(IPEXLinearFusion):
    r"""
    Applies a linear transformation to the `input` data, and then apply PyTorch SILU
    (see https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html)
    on the result:

    .. highlight:: python
    .. code-block:: python

        result = torch.nn.functional.silu(linear(input))

    Args:
        linear (torch.nn.Linear module) : the original torch.nn.Linear
            module to be fused with silu.

    Shape:
        Input and output shapes are the same as torch.nn.Linear.

    Examples:
        >>> # module init:
        >>> linear_module = torch.nn.Linear(4096, 4096)
        >>> ipex_fusion = ipex.llm.modules.LinearSilu(linear_module)
        >>> # module forward:
        >>> input = torch.randn(4096, 4096)
        >>> result = ipex_fusion(input)

    """

    def __init__(self, linear):
        super().__init__(linear)

    def forward(self, x):
        if self.device_type != x.device.type:
            self.init_on_device(x, IPEXCustomOpType.LINEAR_SILU)

        return self.linear_fusion(x)


class Linear2SiluMul(IPEXLinear2Fusion):
    r"""
    Applies two linear transformation to the `input` data (`linear_s` and
    `linear_m`), then apply PyTorch SILU
    (see https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html)
    on the result from `linear_s`, and multiplies the result from `linear_m`:

    .. highlight:: python
    .. code-block:: python

        result = torch.nn.functional.silu(linear_s(input)) * linear_m(input)

    Args:
        linear_s (torch.nn.Linear module) : the original torch.nn.Linear
            module to be fused with silu.
        linear_m (torch.nn.Linear module) : the original torch.nn.Linear
            module to be fused with mul.

    Shape:
        Input and output shapes are the same as torch.nn.Linear.

    Examples:
        >>> # module init:
        >>> linear_s_module = torch.nn.Linear(4096, 4096)
        >>> linear_m_module = torch.nn.Linear(4096, 4096)
        >>> ipex_fusion = ipex.llm.modules.Linear2SiluMul(linear_s_module, linear_m_module)
        >>> # module forward:
        >>> input = torch.randn(4096, 4096)
        >>> result = ipex_fusion(input)

    """

    def __init__(self, linear_s, linear_m):
        super().__init__(linear_s, linear_m)

    def forward(self, x):
        if self.device_type != x.device.type:
            self.init_on_device(x, IPEXCustomOpType.LINEAR2_SILU_MUL)

        return self.linear_fusion(x)


class LinearRelu(IPEXLinearFusion):
    r"""
    Applies a linear transformation to the `input` data, and then apply PyTorch RELU
    (see https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html)
    on the result:

    .. highlight:: python
    .. code-block:: python

        result = torch.nn.functional.relu(linear(input))

    Args:
        linear (torch.nn.Linear module) : the original torch.nn.Linear module
            to be fused with relu.

    Shape:
        Input and output shapes are the same as torch.nn.Linear.

    Examples:
        >>> # module init:
        >>> linear_module = torch.nn.Linear(4096, 4096)
        >>> ipex_fusion = ipex.llm.modules.LinearRelu(linear_module)
        >>> # module forward:
        >>> input = torch.randn(4096, 4096)
        >>> result = ipex_fusion(input)

    """

    def __init__(self, linear):
        super().__init__(linear)

    def forward(self, x):
        if self.device_type != x.device.type:
            self.init_on_device(x, IPEXCustomOpType.LINEAR_RELU)

        return self.linear_fusion(x)


class LinearNewGelu(IPEXLinearFusion):
    r"""
    Applies a linear transformation to the `input` data, and then apply NewGELUActivation
    (see https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L50)
    on the result:

    .. highlight:: python
    .. code-block:: python

        result = NewGELUActivation(linear(input))

    Args:
        linear (torch.nn.Linear module) : the original torch.nn.Linear module
            to be fused with new_gelu.

    Shape:
        Input and output shapes are the same as torch.nn.Linear.

    Examples:
        >>> # module init:
        >>> linear_module = torch.nn.Linear(4096, 4096)
        >>> ipex_fusion = ipex.llm.modules.LinearNewGelu(linear_module)
        >>> # module forward:
        >>> input = torch.randn(4096, 4096)
        >>> result = ipex_fusion(input)

    """

    def __init__(self, linear):
        super().__init__(linear)

    def forward(self, x):
        if self.device_type != x.device.type:
            self.init_on_device(x, IPEXCustomOpType.LINEAR_NEW_GELU)

        return self.linear_fusion(x)


class LinearGelu(IPEXLinearFusion):
    r"""
    Applies a linear transformation to the `input` data, and then apply PyTorch GELU
    (see https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html)
    on the result:

    .. highlight:: python
    .. code-block:: python

        result = torch.nn.functional.gelu(linear(input))

    Args:
        linear (torch.nn.Linear module) : the original torch.nn.Linear
            module to be fused with gelu.

    Shape:
        Input and output shapes are the same as torch.nn.Linear.

    Examples:
        >>> # module init:
        >>> linear_module = torch.nn.Linear(4096, 4096)
        >>> ipex_fusion = ipex.llm.modules.LinearGelu(linear_module)
        >>> # module forward:
        >>> input = torch.randn(4096, 4096)
        >>> result = ipex_fusion(input)

    """

    def __init__(self, linear):
        super().__init__(linear)

    def forward(self, x):
        if self.device_type != x.device.type:
            self.init_on_device(x, IPEXCustomOpType.LINEAR_GELU)

        return self.linear_fusion(x)


class LinearSiluMul(IPEXLinearFusion):
    r"""
    Applies a linear transformation to the `input` data, then apply PyTorch SILU
    (see https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html)
    on the result, and multiplies the result by `other`:

    .. highlight:: python
    .. code-block:: python

        result = torch.nn.functional.silu(linear(input)) * other

    Args:
        linear (torch.nn.Linear module) : the original torch.nn.Linear module to
            be fused with silu and mul.

    Shape:
        Input and output shapes are the same as torch.nn.Linear.

    Examples:
        >>> # module init:
        >>> linear_module = torch.nn.Linear(4096, 4096)
        >>> ipex_fusion = ipex.llm.modules.LinearSiluMul(linear_module)
        >>> # module forward:
        >>> input = torch.randn(4096, 4096)
        >>> other = torch.randn(4096, 4096)
        >>> result = ipex_fusion(input, other)

    """

    def __init__(self, linear):
        super().__init__(linear)

    def forward(self, x, y):
        if self.device_type != x.device.type:
            self.init_on_device(x, IPEXCustomOpType.LINEAR_SILU_MUL)

        return self.linear_fusion(x, y)


class LinearMul(IPEXLinearFusion):
    r"""
    Applies a linear transformation to the `input` data, and then multiplies
    the result by `other`:

    .. highlight:: python
    .. code-block:: python

        result = linear(input) * other

    Args:
        linear (torch.nn.Linear module) : the original torch.nn.Linear module
            to be fused with mul.

    Shape:
        Input and output shapes are the same as torch.nn.Linear.

    Examples:
        >>> # module init:
        >>> linear_module = torch.nn.Linear(4096, 4096)
        >>> ipex_fusion = ipex.llm.modules.LinearMul(linear_module)
        >>> # module forward:
        >>> input = torch.randn(4096, 4096)
        >>> other = torch.randn(4096, 4096)
        >>> result = ipex_fusion(input, other)

    """

    def __init__(self, linear):
        super().__init__(linear)

    def forward(self, x, y):
        if self.device_type != x.device.type:
            self.init_on_device(x, IPEXCustomOpType.LINEAR_MUL)

        return self.linear_fusion(x, y)


class LinearAdd(IPEXLinearFusion):
    r"""
    Applies a linear transformation to the `input` data,
    and then add the result by `other`:

    .. highlight:: python
    .. code-block:: python

        result = linear(input) + other

    Args:
        linear (torch.nn.Linear module) : the original torch.nn.Linear
            module to be fused with add.

    Shape:
        Input and output shapes are the same as torch.nn.Linear.

    Examples:
        >>> # module init:
        >>> linear_module = torch.nn.Linear(4096, 4096)
        >>> ipex_fusion = ipex.llm.modules.LinearAdd(linear_module)
        >>> # module forward:
        >>> input = torch.randn(4096, 4096)
        >>> other = torch.randn(4096, 4096)
        >>> result = ipex_fusion(input, other)

    """

    def __init__(self, linear):
        super().__init__(linear)

    def forward(self, x, y):
        if self.device_type != x.device.type:
            self.init_on_device(x, IPEXCustomOpType.LINEAR_ADD)

        return self.linear_fusion(x, y)


class LinearAddAdd(IPEXLinearFusion):
    r"""
    Applies a linear transformation to the `input` data,
    and then add the result by `other_1` and `other_2`:

    .. highlight:: python
    .. code-block:: python

        result = linear(input) + other_1 + other_2

    Args:
        linear (torch.nn.Linear module) : the original torch.nn.Linear
            module to be fused with add and add.

    Shape:
        Input and output shapes are the same as torch.nn.Linear.

    Examples:
        >>> # module init:
        >>> linear_module = torch.nn.Linear(4096, 4096)
        >>> ipex_fusion = ipex.llm.modules.LinearAddAdd(linear_module)
        >>> # module forward:
        >>> input = torch.randn(4096, 4096)
        >>> other_1 = torch.randn(4096, 4096)
        >>> other_2 = torch.randn(4096, 4096)
        >>> result = ipex_fusion(input, other_1, other_2)

    """

    def __init__(self, linear):
        super().__init__(linear)

    def forward(self, x, y, z):
        if self.device_type != x.device.type:
            self.init_on_device(x, IPEXCustomOpType.LINEAR_ADD_ADD)

        return self.linear_fusion(x, y, z)


class GatedMLPMOE(nn.Module):
    r"""
    GatedMLPMOE layer for MoE models like Mixtral.

    This layer contains both gate_up_proj (W13, or
    gate_proj for W1 and up_proj for W3) and down_proj weights (W2).

    Note: Mixtral uses W1, W2, and W3 for gate, up, and down_proj. We
    copy that naming convention here.

    `module init`

    Args:
        W13 (torch.Tensor): weights tensor of gate_up_proj, or gate_proj only
                            if W3 is provided
        W2 (torch.Tensor): weights tensor of down_proj
        W3 (torch.Tensor): weights tensor of up_proj, default is None if provided
                           with W1, i.e., W13).
        use_prepack (bool): whether to use IPEX weights prepack optimizations or not,
                            default is True.

    `forward()`

    Args:
        hidden_states (torch.Tensor): Input hidden state of the transformer
        use_grouped_topk (bool): whether to use grouped topk instead of topk only
        top_k (int): Number of experts selected for each token
        router_logits (torch.Tensor): The router_logits tensor after gate_proj
        renomalize (bool): Whether to renormalize the logits in the fused_moe kernel
        topk_group (int): The topk group num when using use_grouped_topk
        num_expert_group (int): The expert group num when using use_grouped_topk

    Examples:
        >>> # module init:
        >>> ipex_fusion = ipex.llm.modules.GatedMLPMOE(W13, W2, prepack=True)
        >>> # module forward:
        >>> result = ipex_fusion(hidden_states, False, top_k, router_logits, True)

    """

    def __init__(self, W13, W2, W3=None, use_prepack=True):
        super().__init__()
        self.W13 = W13
        self.W2 = W2
        self.W3 = W3
        self.use_prepack = use_prepack
        self.linear_fusion = None
        self.device_type = None
        self.runtime_ops = IPEXRuntimeCustomOps()

    def init_on_device(self, x, op_type):
        self.device_type = x.device.type
        self.linear_fusion = self.runtime_ops.get_module_from_device(
            self.device_type, op_type, False
        )(self.W13, self.W2, self.W3, self.use_prepack)

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
    ) -> torch.Tensor:
        if self.device_type != hidden_states.device.type:
            self.init_on_device(hidden_states, IPEXCustomOpType.LINEAR_MOE)

        return self.linear_fusion(
            hidden_states,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
        )
