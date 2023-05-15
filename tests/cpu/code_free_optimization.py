import argparse
import torch
import torch.nn as nn


class ConvBatchNorm(torch.nn.Module):
    def __init__(
        self,
    ):
        super(ConvBatchNorm, self).__init__()
        self.conv = torch.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        )
        self.bn = torch.nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBatchNormSoftmax(torch.nn.Module):
    def __init__(
        self,
    ):
        super(ConvBatchNormSoftmax, self).__init__()
        self.conv = torch.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        )
        self.bn = torch.nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

    def forward(self, x):
        return nn.Softmax(dim=-1)(self.bn(self.conv(x)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conv_bn", action="store_true", help="test conv_bn model", default=False
    )
    parser.add_argument(
        "--conv_bn_with_module_created_in_forward",
        action="store_true",
        help="test module created in forward",
        default=False,
    )
    args = parser.parse_args()
    if args.conv_bn:
        input = torch.randn(1, 3, 224, 224)
        model = ConvBatchNorm().eval()
        for i in range(10):
            model(input)
    if args.conv_bn_with_module_created_in_forward:
        input = torch.randn(1, 3, 224, 224)
        model = ConvBatchNormSoftmax().eval()
        for i in range(10):
            model(input)
