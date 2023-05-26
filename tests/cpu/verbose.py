import argparse
import torch
import intel_extension_for_pytorch as ipex


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv = torch.nn.Conv2d(1, 10, 5, 1)

    def forward(self, x):
        y = self.conv(x)
        return y


def run_model(level):
    m = Module().eval()
    m = ipex.optimize(m, dtype=torch.float32, level="O1")
    d = torch.rand(1, 1, 112, 112)
    with ipex.verbose(level):
        m(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", default=0, type=int)
    args = parser.parse_args()
    run_model(args.verbose_level)
