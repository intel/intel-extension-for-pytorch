import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation

from PIL import Image
import requests

cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")


model_dict = [
    # "Image-Segmentation": {
    "mattmdjaga/segformer_b2_clothes",
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "jonathandinu/face-parsing",
    "sayeed99/segformer_b3_clothes",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "nvidia/segformer-b4-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "sayeed99/segformer-b3-fashion",
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    "microsoft/beit-large-finetuned-ade-640-640",
    # }
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

Image_Segmentation_configuration = {
    "mattmdjaga/segformer_b2_clothes": [
        "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80",  # noqa:B950
        SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes"),
        AutoModelForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        ),
    ],
    "nvidia/segformer-b0-finetuned-ade-512-512": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        ),
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        ),
    ],
    "jonathandinu/face-parsing": [
        "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6",
        SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing"),
        SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing"),
    ],
    "sayeed99/segformer_b3_clothes": [
        "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80",  # noqa: B950
        SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes"),
        AutoModelForSemanticSegmentation.from_pretrained(
            "sayeed99/segformer_b3_clothes"
        ),
    ],
    "nvidia/segformer-b2-finetuned-ade-512-512": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        ),
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        ),
    ],
    "nvidia/segformer-b3-finetuned-cityscapes-1024-1024": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
        ),
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
        ),
    ],
    "nvidia/segformer-b4-finetuned-ade-512-512": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b4-finetuned-ade-512-512"
        ),
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b4-finetuned-ade-512-512"
        ),
    ],
    "nvidia/segformer-b3-finetuned-ade-512-512": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b3-finetuned-ade-512-512"
        ),
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-ade-512-512"
        ),
    ],
    "nvidia/segformer-b1-finetuned-cityscapes-1024-1024": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
        ),
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
        ),
    ],
    "sayeed99/segformer-b3-fashion": [
        "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80",  # noqa: B950
        SegformerImageProcessor.from_pretrained("sayeed99/segformer-b3-fashion"),
        AutoModelForSemanticSegmentation.from_pretrained(
            "sayeed99/segformer-b3-fashion"
        ),
    ],
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        ),
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        ),
    ],
    "microsoft/beit-large-finetuned-ade-640-640": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        BeitFeatureExtractor.from_pretrained(
            "microsoft/beit-large-finetuned-ade-640-640"
        ),
        BeitForSemanticSegmentation.from_pretrained(
            "microsoft/beit-large-finetuned-ade-640-640"
        ),
    ],
}


def test_image_segmentation_eval():
    for model_id in model_dict:
        # try:
        url, processor, model = Image_Segmentation_configuration[model_id]
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(images=image, return_tensors="pt")
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

        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        print("-" * 60)
        print(
            "Testing image_segmentation for model: {}, result device:{}, result:{}".format(
                model_id, model.device.type, pred_seg
            )
        )
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_image_segmentation_eval()
