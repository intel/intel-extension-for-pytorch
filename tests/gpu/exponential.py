import torch
import matplotlib.pyplot as plt
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")
exp_dist = torch.ones(1000000, device=dpcpp_device)
exp_dist.exponential_(1)

print("exponential device ", exp_dist.device)
print("exponential ", exp_dist.to("cpu"))

np_data = exp_dist.cpu().detach().numpy()

print("numpy ", np_data)
plt.hist(np_data, 100)
plt.show()


