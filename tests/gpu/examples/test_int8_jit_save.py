import torch
import torch.nn as nn
import copy
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

from torch.quantization.quantize_jit import (
    convert_jit,
    prepare_jit,
)


class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        ic = 17
        c = 18
        self.block = nn.Sequential(
            nn.Conv2d(3, ic, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ic, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.fc = nn.Linear(c * 64, 256)

    def forward(self, x):
        res1 = self.block(x)
        res1 = res1.view(res1.size(0), -1)
        res1 = self.fc(res1)
        return res1


class ConvBias(torch.nn.Module):
    def __init__(self):
        super(ConvBias, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )
        print(self.block[0].bias)

    def forward(self, x):
        return self.block(x)
        # return x


class LinearBias(torch.nn.Module):
    def __init__(self):
        super(LinearBias, self).__init__()
        self.fc = nn.Linear(128, 3)
        print(self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        return x


def trace_int8_model(model, device, test_input):
    model = model.to(device)
    modelJit = torch.jit.script(model)
    # modelJit = torch.jit.trace(model, test_input.to("cpu"))
    modelJit.eval()
    modelJit.to(device)
    print(modelJit)
    print("finish jit scripting...")

    print("start ", device, " calibration ...")
    qconfig_u8 = torch.quantization.QConfig(
        activation=torch.quantization.observer.MinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric, reduce_range=False, dtype=torch.quint8
        ),
        weight=torch.quantization.default_weight_observer,
    )

    modelJit = prepare_jit(modelJit, {"": qconfig_u8}, True)

    # do calibration
    test_input = test_input.to(device)
    with torch.no_grad():
        for i in range(1):
            calib_input = test_input
            modelJit(calib_input)
    print("start ", device, " convert...")
    modelJit = convert_jit(modelJit, True)
    # inference
    print("start ", device, " inference ...")
    with torch.inference_mode():
        for i in range(1):
            print("inference iter:", i)
            test_res = modelJit(test_input.to(device))
    return modelJit, test_res


class TestTorchMethod(TestCase):
    def test_composite(self, dtype=torch.float):
        for device in ["xpu", "cpu"]:
            for model in [LinearBias(), ConvBias()]:
                if device == "xpu":
                    torch.backends.quantized.engine = "qxpu"
                    # torch.xpu.set_quant_save_backend("xpu")
                else:
                    torch.backends.quantized.engine = "x86"
                if isinstance(model, LinearBias):
                    test_input = torch.randn([4, 128])
                else:
                    test_input = torch.rand([1, 3, 8, 8])

                modelJit = copy.deepcopy(model)
                modelJit, xpu_res = trace_int8_model(modelJit, device, test_input)
                print("===== start saving model =====")
                torch.jit.save(modelJit, "int8_conv_sigmoid.pt")
                # print(modelJit.inlined_graph)
                print("===== start loading model =====")
                modelLoad = torch.jit.load("int8_conv_sigmoid.pt")
                print("===== end loading model =====")
                # print(modelLoad.inlined_graph)
                # with torch.inference_mode():
                print("start test inference")
                # modelJit = modelJit.to("cpu")
                # modelLoad = modelLoad.to("cpu")
                with torch.no_grad():
                    xpu_res = modelJit(test_input.to(device))
                    load_res = modelLoad(test_input.to(device))
                print("end test inference")
                print(xpu_res.dtype)
                print(load_res.dtype)
                self.assertEqual(load_res.cpu(), xpu_res.cpu())
        torch.backends.quantized.engine = "x86"  # set back to default engine

    def test_imperative(self):
        test_input = torch.rand([1, 3, 8, 8])
        model = ConvBias()
        model = model.to("xpu")
        model1 = copy.deepcopy(model)
        model = torch.quantization.QuantWrapper(model)
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.observer.MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric),
            weight=torch.quantization.default_weight_observer)
        model.qconfig = qconfig

        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        print(model(test_input.to("xpu"))[0][0:5])
        # model1 = copy.deepcopy(model)

        torch.save(model.state_dict(), "impe.pt")
        wgh_file = torch.load("impe.pt")
        model.load_state_dict(wgh_file)

        for o, l in zip(model.parameters(), model1.parameters()):
            print(f"o: {o.cpu()}")
            print(f"l: {l.cpu()}")
            assert torch.equal(o, l), " param tensor in saved & loaded not equal"
