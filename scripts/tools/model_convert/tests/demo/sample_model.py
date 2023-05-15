import torch
import torch.cuda

torch.set_default_tensor_type(torch.cuda.DoubleTensor)


class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3, 3)
        self.relu1 = torch.nn.ReLU(3)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        y = y + 1
        return x, y


def run():
    model = SampleModel()
    input1 = torch.randn(3, 3).cuda()
    input2 = torch.randn(3, 3, device="cuda:0", dtype=torch.float)
    outputs = model(input1, input2)
    print("CUDA model run succeed.")


if __name__ == "__main__":
    run()
