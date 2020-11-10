import numpy
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
	def test_TensorFactories(self, dtype=torch.float):

		x = torch.empty_strided((2, 3), (1, 2), device=cpu_device)
		x_out = torch.empty_strided((2, 3), (1, 2), device=dpcpp_device)
		y = torch.eye(3, device=cpu_device)
		y_out = torch.eye(3, device=dpcpp_device)
		m = torch.tril_indices(3, 3, device=cpu_device)
		n = torch.triu_indices(3, 3, device=cpu_device)
		m_out = torch.tril_indices(3, 3, device=dpcpp_device)
		n_out = torch.triu_indices(3, 3, device=dpcpp_device)

		print("cpu: ")
		print(x)
		print(x.stride())
		print(x.size())
		print(y)
		print(m)
		print(n)

		print("dpcpp: ")
		print(x_out.to("cpu"))
		print(x_out.stride())
		print(x_out.size())
		print(y_out.to("cpu"))
		print(m_out.to("cpu"))
		print(n_out.to("cpu"))
		self.assertEqual(x.size(), x_out.size())
		self.assertEqual(x.stride(), x_out.stride())
		self.assertEqual(y, y_out.cpu())
		self.assertEqual(m, m_out.cpu())
		self.assertEqual(n, n_out.cpu())
