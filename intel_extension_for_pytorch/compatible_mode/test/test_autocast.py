import torch
import urllib
import intel_extension_for_pytorch as ipex

from PIL import Image
from torchvision import transforms

ipex.compatible_mode()

model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained=True)
model.eval()

if torch.cuda.is_available():
    model.to("cuda")


url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "dog.jpg",
)  # noqa: E501
try:
    urllib.URLopener().retrieve(url, filename)
except Exception:
    urllib.request.urlretrieve(url, filename)


input_image = Image.open(filename)
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # noqa: E501
    ]
)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

scaler = torch.cuda.amp.GradScaler()

# with torch.cuda.amp.autocast():
#    output = model(input_batch)

with torch.autocast(device_type="cuda"):
    output = model(input_batch)

print(output[0])
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
