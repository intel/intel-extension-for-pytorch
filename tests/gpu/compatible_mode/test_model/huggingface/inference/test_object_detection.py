import torch
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

from PIL import Image
import requests

from transformers import AutoImageProcessor
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    DetrFeatureExtractor,
)
from transformers import (
    YolosImageProcessor,
    YolosForObjectDetection,
    YolosFeatureExtractor,
)
from transformers import (
    DeformableDetrForObjectDetection,
    ConditionalDetrForObjectDetection,
)
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model_dict = [
    # "Object-Detection": {
    "facebook/detr-resnet-101",  # Failed to skip
    "facebook/detr-resnet-50",  # Failed to skip
    "hustvl/yolos-tiny",
    "SenseTime/deformable-detr",
    "Aryn/deformable-detr-DocLayNet",
    "nickmuchi/yolos-small-finetuned-license-plate-detection",
    "SenseTime/deformable-detr-with-box-refine",
    "hustvl/yolos-base",
    "microsoft/conditional-detr-resnet-50",
    "PekingU/rtdetr_r50vd",
    "facebook/detr-resnet-101-dc5",
    "PekingU/rtdetr_r101vd_coco_o365",
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

object_detection_configutation = {
    "facebook/detr-resnet-101": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-101", revision="no_timm"
        ),
        DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101", revision="no_timm"
        ),
    ],
    "facebook/detr-resnet-50": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        ),
        DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        ),
    ],
    "hustvl/yolos-tiny": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        YolosImageProcessor.from_pretrained("hustvl/yolos-tiny"),
        YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny"),
    ],
    "SenseTime/deformable-detr": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("SenseTime/deformable-detr"),
        DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr"),
    ],
    "Aryn/deformable-detr-DocLayNet": [
        "https://huggingface.co/Aryn/deformable-detr-DocLayNet/resolve/main/examples/doclaynet_example_1.png",
        AutoImageProcessor.from_pretrained("Aryn/deformable-detr-DocLayNet"),
        DeformableDetrForObjectDetection.from_pretrained(
            "Aryn/deformable-detr-DocLayNet"
        ),
    ],
    "nickmuchi/yolos-small-finetuned-license-plate-detection": [
        "https://drive.google.com/uc?id=1p9wJIqRz3W50e2f_A0D8ftla8hoXz4T5",
        YolosFeatureExtractor.from_pretrained(
            "nickmuchi/yolos-small-finetuned-license-plate-detection"
        ),
        YolosForObjectDetection.from_pretrained(
            "nickmuchi/yolos-small-finetuned-license-plate-detection"
        ),
    ],
    "SenseTime/deformable-detr-with-box-refine": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("SenseTime/deformable-detr-with-box-refine"),
        DeformableDetrForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr-with-box-refine"
        ),
    ],
    "hustvl/yolos-base": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        YolosFeatureExtractor.from_pretrained("hustvl/yolos-base"),
        YolosForObjectDetection.from_pretrained("hustvl/yolos-base"),
    ],
    "microsoft/conditional-detr-resnet-50": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50"),
        ConditionalDetrForObjectDetection.from_pretrained(
            "microsoft/conditional-detr-resnet-50"
        ),
    ],
    "PekingU/rtdetr_r50vd": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd"),
        RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd"),
    ],
    "facebook/detr-resnet-101-dc5": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101-dc5"),
        DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101-dc5"),
    ],
    "PekingU/rtdetr_r101vd_coco_o365": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd_coco_o365"),
        RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r101vd_coco_o365"),
    ],
}


def test_object_detection_eval():
    for model_id in model_dict:
        # try:
        url, processor, model = object_detection_configutation[model_id]
        image = Image.open(requests.get(url, stream=True).raw)
        if args.precision == "fp16":
            inputs = processor(
                images=image,
                return_tensors="pt",
                device=cuda_device,
                dtype=torch.float16,
            )
            model = model.to(cuda_device, dtype=torch.float16)
        elif args.precision == "bf16":
            inputs = processor(
                images=image,
                return_tensors="pt",
                device=cuda_device,
                dtype=torch.bfloat16,
            )
            model = model.to(cuda_device, dtype=torch.bfloat16)
        else:
            inputs = processor(images=image, return_tensors="pt", device=cuda_device)
            model = model.to(cuda_device)
        if args.backend == "torch_compile":
            model = torch.compile(model)

        for k in inputs.keys():
            inputs[k] = inputs[k].to(cuda_device)

        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        print("-" * 60)
        print(
            "Testing model name:{}, result device:{}".format(
                model_id, results["scores"].device.type
            )
        )
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "  # noqa E126
                f"{round(score.item(), 3)} at location {box}"
            )
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_object_detection_eval()
