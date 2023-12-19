import torch  # noqa
from torch.testing._internal.common_utils import TestCase
from torch import nn
from copy import deepcopy
import intel_extension_for_pytorch  # noqa
import math

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class test_module(nn.Module):
    def __init__(self):
        super(test_module, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(2, 2)

    def pass_forward(self, X_data):
        X_data = self.pool(F.relu(self.conv1(X_data)))
        X_data = self.pool(F.relu(self.conv2(X_data)))
        X_data = X_data.view(-1, 16 * 5 * 5)
        X_data = F.relu(self.fc1(X_data))
        X_data = F.relu(self.fc2(X_data))
        X_data = self.fc3(X_data)

        return X_data


class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        """Create EMA."""
        self.ema = deepcopy(model)  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / 2000)
        )  # decay exponential ramp (to help early epochs)

    def update(self, model):
        """Update EMA parameters."""
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.state_dict()  # model state_dict
        # print(self.ema.state_dict())
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
                # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'
        return self.ema.state_dict()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates attributes and saves stripped model with optimizer removed."""
        copy_attr(self.ema, model, include, exclude)


class TestTorchMethod(TestCase):
    def test_ema(self, dtype=torch.float):
        net = test_module().to(dpcpp_device)
        decay = 0.999

        ipex_ema = torch.xpu.EMA(net, decay=decay)
        python_ema = ModelEMA(net, decay=decay)

        python_res = python_ema.update(net)
        kernel_res = ipex_ema.update(net)

        print("python res:", python_res, "\n")
        print("kernel res:", kernel_res, "\n")
        self.assertEqual(python_res, kernel_res)
