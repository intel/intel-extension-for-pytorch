import torch

if torch.cuda.is_available():
    print("cuda is available")
    if torch.cuda.get_device_capability(torch.cuda.current_device()) == (8, 0):
        print("8.0" is supported)
else:
    print("cuda is not available")
