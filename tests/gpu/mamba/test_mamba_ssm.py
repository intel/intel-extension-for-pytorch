# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import random
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat

import intel_extension_for_pytorch as ipex

def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def selective_scan_ref(u,
                       delta,
                       A,
                       B,
                       C,
                       D=None,
                       z=None,
                       delta_bias=None,
                       delta_softplus=False,
                       return_last_state=False,
                       prev_state=None,
                       final_state_out=None):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    prev_state: r(B D N), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate)) if prev_state is None else prev_state
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            if final_state_out is None:
                final_state_out = x
            else:
                final_state_out.copy_(x)
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, final_state_out)


@pytest.mark.parametrize('wtype', [torch.float32])
@pytest.mark.parametrize('itype',
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize('has_delta_bias', [True])
@pytest.mark.parametrize('delta_softplus', [True])
@pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("is_variable_C", [True])
@pytest.mark.parametrize("is_variable_B", [True])
@pytest.mark.parametrize("scan_chunks", [1])
def test_selective_scan(is_variable_B, is_variable_C, varBC_groups,
                        has_z, has_delta_bias, delta_softplus, seqlen, itype,
                        wtype, scan_chunks):
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = 'xpu'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    seed_everything(0)
    batch_size = 1
    dim = 4
    dstate = 1
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype))
    A_ref = A.clone()
    if not is_variable_B:
        B_shape = [dim, dstate]
    elif varBC_groups == 1:
        B_shape = [batch_size, dstate, seqlen]
    else:
        B_shape = [batch_size, varBC_groups, dstate, seqlen]
    B = torch.randn(B_shape,
                    device=device,
                    dtype=wtype if not is_variable_B else itype)
    B_ref = B.clone()
    if not is_variable_C:
        C_shape = [dim, dstate]
    elif varBC_groups == 1:
        C_shape = [batch_size, dstate, seqlen]
    else:
        C_shape = [batch_size, varBC_groups, dstate, seqlen]
    C = torch.randn(C_shape,
                    device=device,
                    dtype=wtype if not is_variable_C else itype)
    C_ref = C.clone()
    D = torch.randn(dim, device=device, dtype=torch.float32)
    D_ref = D.clone()
    z = torch.randn(batch_size, dim, seqlen, device=device,
                    dtype=itype) if has_z else None
    z_ref = z.clone() if has_z else None
    delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)
                  ) if has_delta_bias else None
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
    u_ref = u.clone()
    delta = (0.5 *
             torch.rand(batch_size, dim, seqlen, device=device, dtype=itype))
    delta_ref = delta.clone()
    state_shape = (batch_size, u.shape[1], int(A.shape[1]))
    state = torch.randn(state_shape,
                        device=u.device,
                        dtype=itype,
                        requires_grad=False)
    state_ref = state.clone()
    out = None
    out_ref = None
    outs = []
    for c in range(scan_chunks):
        chunked_prompt_len = seqlen // scan_chunks
        chunk_start = chunked_prompt_len * c
        chunk_end = chunked_prompt_len * (c + 1)
        if c == scan_chunks - 1:
            chunk_end = seqlen
        _B = B
        if is_variable_B:
            _B = B[..., chunk_start:chunk_end]
        _C = C
        if is_variable_B:
            _C = C[..., chunk_start:chunk_end]
        _z = z
        if has_z:
            assert z is not None
            _z = z[..., chunk_start:chunk_end]
        out, state = ipex.llm.modules.MambaMixer.selective_scan_fn(
            u[..., chunk_start:chunk_end],
            delta[..., chunk_start:chunk_end],
            A,
            _B,
            _C,
            D,
            z=_z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=True)
        outs.append(out)
    if len(outs) > 1:
        out = torch.cat(outs, dim=-1)

    # print(state_0.shape, state.shape)

    out_ref, state_ref, *rest = selective_scan_ref(
        u_ref,
        delta_ref,
        A_ref,
        B_ref,
        C_ref,
        D_ref,
        z=z_ref,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=True)

    assert out is not None and out_ref is not None
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    assert state is not None and state_ref is not None
    assert torch.allclose(state, state_ref.to(itype), rtol=rtol, atol=atol)

