import torch
import torchvision.models as models
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert

# construct the model
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
qconfig = ipex.quantization.default_static_qconfig
model.eval()
example_inputs = torch.rand(1, 3, 224, 224)
prepared_model = prepare(model, qconfig, example_inputs=example_inputs, inplace=False)

##### Example Dataloader #####  # noqa F401
import torchvision
DOWNLOAD = True
DATA = 'datasets/cifar10/'

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA,
    train=True,
    transform=transform,
    download=DOWNLOAD,
)
calibration_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=128
)

with torch.no_grad():
    for batch_idx, (d, target) in enumerate(calibration_data_loader):
        print(f'calibrated on batch {batch_idx} out of {len(calibration_data_loader)}')
        prepared_model(d)
##############################  # noqa F401

convert_model = convert(prepared_model)
with torch.no_grad():
    traced_model = torch.jit.trace(convert_model, example_inputs)
    traced_model = torch.jit.freeze(traced_model)

traced_model.save("quantized_model.pt")

# Deployment
quantized_model = torch.jit.load("quantized_model.pt")
quantized_model = torch.jit.freeze(quantized_model.eval())
images = torch.rand(1, 3, 244, 244)
with torch.no_grad():
    output = quantized_model(images)

print("Execution finished")
