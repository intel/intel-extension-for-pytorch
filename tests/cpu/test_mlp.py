import torch
import time
import intel_pytorch_extension as ipex
K=1 #128
C=16 #64
MB = 28

def get_rand_seed():
    return int(time.time() * 1000000000)

def _ipxex_linear(random_seed, data_type = torch.float32):
  torch.manual_seed(random_seed)
  fc = ipex.IpexMLPLinear(C, K).to(data_type)
  return fc

def _cpu_linear(random_seed, data_type = torch.float32):
  torch.manual_seed(random_seed)
  fc = torch.nn.Linear(C, K).to(data_type)
  return fc

def _run_mlp(random_seed, fc_module, data_type = torch.float32):
  torch.manual_seed(random_seed)
  x1 = torch.randn(MB, C, requires_grad=True).to(data_type).requires_grad_(True)
  y1 = fc_module(x1)
  z1 = y1.mean()
  z1.backward()
  return x1.grad, fc_module.weight.grad, fc_module.bias.grad

for data_type in [torch.float32, torch.bfloat16]:
  seed = get_rand_seed()
  ipex_fc = _ipxex_linear(seed, data_type)
  cpu_fc = _cpu_linear(seed, data_type)

  rtol = 1e-5
  atol = rtol
  if data_type == torch.bfloat16:
    rtol = 1e-2
    atol = rtol

  seed = get_rand_seed()
  input_grad_ipex, weight_grad_ipex, bias_grad_ipex = _run_mlp(seed, ipex_fc, data_type)
  input_grad_cpu, weight_grad_cpu, bias_grad_cpu = _run_mlp(seed, cpu_fc, data_type)

  if input_grad_ipex is None:
    if input_grad_cpu is not None:
      print("##################### {} MLP input grad FAIL".format(str(data_type)))
    else:
      print("##################### {} MLP input grad PASS".format(str(data_type)))
  else:
    if not input_grad_ipex.to(torch.float32).allclose(input_grad_cpu.to(torch.float32), rtol=rtol, atol=atol):
      print("##################### {} MLP input grad FAIL".format(str(data_type)))
    else:
      print("##################### {} MLP input grad PASS".format(str(data_type)))

  if not weight_grad_ipex.to(torch.float32).allclose(weight_grad_cpu.to(torch.float32), rtol=rtol, atol=atol):
    print("##################### {} MLP weight grad FAIL".format(str(data_type)))
  else:
    print("##################### {} MLP weight grad PASS".format(str(data_type)))

  if not bias_grad_ipex.to(torch.float32).allclose(bias_grad_cpu.to(torch.float32), rtol=rtol, atol=atol):
    print("##################### {} MLP bias grad FAIL".format(str(data_type)))
  else:
    print("##################### {} MLP bias grad PASS".format(str(data_type)))
