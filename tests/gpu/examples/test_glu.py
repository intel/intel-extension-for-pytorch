import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD
from torch.testing._internal.common_utils import TestCase
from typing import Tuple
from torch.overrides import is_tensor_like
from itertools import product

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


def _is_float_or_complex_tensor(obj):
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())


def _allocate_jacobians_with_outputs(
    output_tensors: Tuple, numel_input, dtype=None, device=None
) -> Tuple[torch.Tensor, ...]:
    # Makes zero-filled tensors from outputs. If `dim` is not None, for each tensor
    # in `output_tensors`, returns a new zero-filled tensor with height of `dim` and
    # width of `t.numel`. Otherwise, for each tensor, returns a 1-d tensor with size
    # (t.numel,).
    out: List[torch.Tensor] = []
    options = {"dtype": dtype, "device": device, "layout": torch.strided}
    for t in output_tensors:
        if _is_float_or_complex_tensor(t):
            out.append(t.new_zeros((numel_input, t.numel()), **options))
    return tuple(out)


def _get_analytical_jacobian_forward_ad(
    fn, inputs, outputs
) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    tensor_inputs = tuple(i for i in inputs if is_tensor_like(i) and i.requires_grad)

    if any(i.is_complex() for i in tensor_inputs):
        raise ValueError(
            "Expected inputs to be non-complex for _get_analytical_jacobian_forward_ad."
        )

    jacobians = tuple(
        _allocate_jacobians_with_outputs(outputs, i.numel()) for i in tensor_inputs
    )

    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        for i, inp in enumerate(inputs):
            if is_tensor_like(inp) and inp.requires_grad:
                if inp.layout == torch._mkldnn:  # type: ignore[attr-defined]
                    raise ValueError(
                        "MKLDNN inputs are not support for forward AD gradcheck."
                    )

                inp = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
                # If inp is a differentiable view, the dual might not be the tangent given to
                # make_dual, so read it explicitly from the dual tensor
                fw_grads.append(fwAD.unpack_dual(inp)[1])
            dual_inputs.append(inp)

        # Reconstruct the full Jacobian column by column
        for i, fw_grad in enumerate(fw_grads):
            for lin_idx, grad_idx in enumerate(
                product(*[range(m) for m in fw_grad.size()])
            ):
                fw_grad[grad_idx] = 1.0
                raw_outputs = _as_tuple(fn(*dual_inputs))
                dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
                for index_o, d_o in enumerate(dual_outputs):
                    val, res = fwAD.unpack_dual(d_o)

                    if res is None:
                        jacobians[i][index_o][lin_idx].zero_()
                    else:
                        jacobians[i][index_o][lin_idx].copy_(res.reshape(-1))
                fw_grad[grad_idx] = 0.0

    return jacobians


def _get_analytical_jacobian_forward_ad_backward(
    func, inputs, tupled_grad_outputs
) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    tupled_inputs = _as_tuple(inputs)

    num_outputs = len(tupled_grad_outputs)

    diff_input_args_indices = {
        i for i, x in enumerate(tupled_inputs) if is_tensor_like(x) and x.requires_grad
    }
    diff_grad_output_indices = {
        i for i, x in enumerate(tupled_grad_outputs) if x.requires_grad
    }

    def new_func(*args):
        # Restore the requires_grad information
        input_args = tuple(
            x.requires_grad_() if i in diff_input_args_indices else x
            for i, x in enumerate(args[:-num_outputs])
        )
        outputs = _differentiable_outputs(func(*input_args))
        grad_outputs = tuple(
            x.requires_grad_() if i in diff_grad_output_indices else x
            for i, x in enumerate(args[-num_outputs:])
        )
        diff_input_args = tuple(
            x for i, x in enumerate(input_args) if i in diff_input_args_indices
        )
        grad_inputs = torch.autograd.grad(
            outputs, diff_input_args, grad_outputs, create_graph=True, allow_unused=True
        )
        grad_inputs = tuple(g for g in grad_inputs if g is not None)
        return grad_inputs

    new_inputs = tupled_inputs + tupled_grad_outputs
    return _get_analytical_jacobian_forward_ad(
        new_func, new_inputs, _as_tuple(new_func(*new_inputs))
    )


class TestNNMethod(TestCase):
    def test_glu(self, dtype=torch.float):
        input_cpu = torch.randn(4, 6)
        input_dpcpp = input_cpu.to("xpu")
        m = nn.GLU()

        input_cpu.requires_grad = True
        output_cpu = m(input_cpu)
        output_cpu.backward(torch.ones_like(output_cpu))

        input_dpcpp.requires_grad = True
        output_dpcpp = m(input_dpcpp)
        output_dpcpp.backward(torch.ones_like(output_dpcpp).to("xpu"))
        self.assertEqual(output_cpu, output_dpcpp)
        self.assertEqual(input_cpu.grad, input_dpcpp.grad)

    def test_glu_jvp(self, dtype=torch.float):
        input_cpu = torch.randn(12, 16, requires_grad=True)
        input_xpu = input_cpu.to("xpu").requires_grad_(True)
        model = nn.GLU()

        cpu_out = _get_analytical_jacobian_forward_ad(
            model, _as_tuple(input_cpu), _as_tuple(model(input_cpu))
        )
        xpu_out = _get_analytical_jacobian_forward_ad(
            model, _as_tuple(input_xpu), _as_tuple(model(input_xpu))
        )
        self.assertEqual(cpu_out[0][0], xpu_out[0][0].cpu())

        outputs = _differentiable_outputs(model(input_cpu))
        tupled_grad_outputs = tuple(
            torch.testing.make_tensor(
                x.shape,
                dtype=x.dtype
                if x.is_floating_point() or x.is_complex()
                else torch.double,
                device="cpu",
                low=-1,
                high=1,
                requires_grad=True,
                noncontiguous=False,
            )
            for x in outputs
        )
        tupled_grad_outputs_xpu = tuple(x.to("xpu") for x in tupled_grad_outputs)

        test_cpu = _get_analytical_jacobian_forward_ad_backward(
            model, input_cpu, tupled_grad_outputs
        )
        test_xpu = _get_analytical_jacobian_forward_ad_backward(
            model, input_xpu, tupled_grad_outputs_xpu
        )
        self.assertEqual(cpu_out[0][0], xpu_out[0][0].cpu())
