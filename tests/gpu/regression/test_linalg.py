import pytest
import socket
import torch
import intel_extension_for_pytorch

@pytest.mark.parametrize('prec', [torch.float32, torch.float64])
def test_slogdet(prec):
    device = torch.device('xpu')
    bs, N = 128, 4
    shape = (bs, N, N)

    A = torch.eye(N, dtype=prec, device=device).broadcast_to(*shape) \
    + torch.rand(*shape, dtype=prec, device=device) \
    + 1j*torch.rand(*shape, dtype=prec, device=device)

    s, ldj = torch.linalg.slogdet(A)
    answ_s, answ_ldj = torch.linalg.slogdet(A.cpu())

    assert torch.allclose(s.cpu(), answ_s)
    assert torch.allclose(ldj.cpu(), answ_ldj)