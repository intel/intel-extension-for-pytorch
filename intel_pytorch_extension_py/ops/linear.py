import torch
from torch.autograd import Function
import torch.nn.functional as F
import _torch_ipex as core

F.linear = torch.ops.torch_ipex.linear
