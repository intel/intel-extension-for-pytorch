import torch
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()
from copy import deepcopy

from PIL import Image
import requests
import numpy as np

from transformers import DPTImageProcessor, DPTForDepthEstimation
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import GLPNImageProcessor, GLPNForDepthEstimation


cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model_dict = [
    # "Depth-Estimation": {
    "ProsusAI/finbert",
    "LiheYoung/depth-anything-small-hf",
    "vinvino02/glpn-kitti",
    "facebook/dpt-dinov2-base-kitti",
    "depth-anything/Depth-Anything-V2-Small-hf",
    "LiheYoung/depth-anything-large-hf",
    "LiheYoung/depth-anything-base-hf",
    "vinvino02/glpn-nyu",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "facebook/dpt-dinov2-large-nyu",
    "facebook/dpt-dinov2-small-kitti",
    # },
]
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="modelname")
    parser.add_argument("--precision", type=str, help="precision")
    parser.add_argument("--backend", type=str, help="backend, torch.compile or eager")
    return parser.parse_args()


args = parse_arguments()
if args.model:
    model_dict = [args.model]


depth_extimation_configuration = {
    "ProsusAI/finbert": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas"),
        DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas", low_cpu_mem_usage=True
        ),
    ],
    "LiheYoung/depth-anything-small-hf": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf"),
        AutoModelForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-small-hf"
        ),
    ],
    "vinvino02/glpn-kitti": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        GLPNImageProcessor.from_pretrained("vinvino02/glpn-kitti"),
        GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti"),
    ],
    "facebook/dpt-dinov2-base-kitti": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-base-kitti"),
        DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-base-kitti"),
    ],
    "Intel/zoedepth-nyu": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
    ],
    "depth-anything/Depth-Anything-V2-Small-hf": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf"),
        AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        ),
    ],
    "LiheYoung/depth-anything-large-hf": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf"),
        AutoModelForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-large-hf"
        ),
    ],
    "LiheYoung/depth-anything-base-hf": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-base-hf"),
        AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-base-hf"),
    ],
    "vinvino02/glpn-nyu": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu"),
        GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu"),
    ],
    "depth-anything/Depth-Anything-V2-Base-hf": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf"),
        AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Base-hf"
        ),
    ],
    "facebook/dpt-dinov2-large-nyu": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-large-nyu"),
        DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-large-nyu"),
    ],
    "facebook/dpt-dinov2-small-kitti": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-small-kitti"),
        DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-small-kitti"),
    ],
}


def test_once(model_id):
    url, image_processor, model = depth_extimation_configuration[model_id]

    image = Image.open(requests.get(url, stream=True).raw)
    inputs = image_processor(images=image, return_tensors="pt")
    if args.precision == "fp16":
        inputs = inputs.to(device=cuda_device, dtype=torch.float16)
        model = model.to(cuda_device, dtype=torch.float16)
    elif args.precision == "bf16":
        inputs = inputs.to(device=cuda_device, dtype=torch.bfloat16)
        model = model.to(cuda_device, dtype=torch.bfloat16)
    else:
        inputs = inputs.to(cuda_device)
        model = model.to(cuda_device)
    if args.backend == "torch_compile":
        model = torch.compile(model)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    running_device = deepcopy(prediction.device)

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    print("*" * 60)
    print("testing model:", model_id)
    print("testing device:", running_device)


def test_depth_extimation_eval():
    for model_id in model_dict:
        test_once(model_id)
        print(f"Testing model success: {model_id}")


if __name__ == "__main__":
    test_depth_extimation_eval()
