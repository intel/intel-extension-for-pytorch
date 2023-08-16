import argparse
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex


class TestModel(torch.nn.Module):
    def __init__(self, ic, oc, bias):
        super(TestModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=bias)
        self.linear = nn.Linear(ic, oc, bias=bias)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = torch.matmul(y1, torch.transpose(y1, 0, 1))
        y3 = self.linear(y2)
        return y3

class TestLSTM(torch.nn.Module):
    def __init__(self):
        super(TestLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=1024, hidden_size=1024)

    def forward(self, x):
        y,_ = self.lstm(x)
        return y

def run_model(args):
    mode = args.mode
    fpmath_mode = ipex.FP32MathMode.BF32 if args.fpmath == "BF32" else ipex.FP32MathMode.FP32
    if not args.env:
        ipex.set_fp32_math_mode(mode=fpmath_mode, device="cpu")
    inputs = torch.randn(1, 3, 224, 224).requires_grad_()
    inputs2 = torch.randn(50, 50, 1024).requires_grad_()
    if args.bias:
        model = TestModel(112, 10, True).eval()
    else:
        model = TestModel(112, 10, False).eval()
    model2 = TestLSTM().train()
    model = ipex.optimize(model, dtype=torch.float32, level='O1', auto_kernel_selection=True)
    ipex.nn.utils._model_convert.replace_lstm_with_ipex_lstm(model2, None)
    if mode == "jit":
        model = torch.jit.trace(model, inputs).eval()
        model = torch.jit.freeze(model)
    output = model(inputs)
    output2 = model2(inputs2)
    output.sum().backward()
    output2.sum().backward()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="imperative", type=str)
    parser.add_argument("--fpmath", default="BF32", type=str)
    parser.add_argument("--env", action='store_true', default=False)
    parser.add_argument("--bias", default=False, type=bool)
    args = parser.parse_args()
    run_model(args)
