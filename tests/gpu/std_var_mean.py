import torch
import torch_ipex

input_cpu = torch.randn(1, 3, dtype=torch.float32, device=torch.device("cpu"))
output_cpu = torch.var_mean(input_cpu)
input_dpcpp = input_cpu.to("dpcpp")
output_dpcpp = torch.var_mean(input_dpcpp)
print(output_cpu[0])
print(output_dpcpp[0])
print(output_cpu[1])
print(output_dpcpp[1])


input_cpu = torch.randn(4, 4, dtype=torch.float32, device=torch.device("cpu"))
output_cpu = torch.var_mean(input_cpu, 1)
input_dpcpp = input_cpu.to("dpcpp")
output_dpcpp = torch.var_mean(input_dpcpp, 1)
print(output_cpu[0])
print(output_dpcpp[0])
print(output_cpu[1])
print(output_dpcpp[1])

input_cpu = torch.randn(1, 3, dtype=torch.float32, device=torch.device("cpu"))
output_cpu = torch.std_mean(input_cpu)
input_dpcpp = input_cpu.to("dpcpp")
output_dpcpp = torch.std_mean(input_dpcpp)
print(output_cpu[0])
print(output_dpcpp[0])
print(output_cpu[1])
print(output_dpcpp[1])