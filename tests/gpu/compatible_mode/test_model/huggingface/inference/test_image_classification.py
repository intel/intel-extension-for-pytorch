import torch
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")


from datasets import load_dataset

from transformers import (
    AutoImageProcessor,
    ResNetForImageClassification,
    BeitFeatureExtractor,
    EfficientNetImageProcessor,
)
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoModelForImageClassification
from transformers import ViTFeatureExtractor
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import SegformerImageProcessor, SegformerForImageClassification
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
from transformers import CvtForImageClassification, EfficientNetForImageClassification

model_dict = [
    # "Image-Classification": {
    "microsoft/resnet-50",
    "google/vit-base-patch16-224",
    "Falconsai/nsfw_image_detection",
    "nateraw/vit-age-classifier",
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "microsoft/resnet-18",
    "nvidia/mit-b0",
    "google/vit-base-patch16-384",
    "google/vit-large-patch16-224",
    "facebook/deit-base-distilled-patch16-224",
    "google/vit-base-patch32-384",
    "microsoft/resnet-101",
    "microsoft/cvt-13",
    "google/vit-large-patch16-384",
    "microsoft/beit-base-patch16-224",
    "microsoft/beit-large-patch16-512",
    "microsoft/resnet-152",
    "google/mobilenet_v2_1.0_224",
    "google/efficientnet-b7",
    "facebook/convnext-base-224",
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

image_classification_configutation = {
    "microsoft/resnet-50": [
        AutoImageProcessor.from_pretrained("microsoft/resnet-50"),
        ResNetForImageClassification.from_pretrained("microsoft/resnet-50"),
    ],
    "google/vit-base-patch16-224": [
        ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224", trust_remote_code=True
        ),
        ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", trust_remote_code=True
        ),
    ],
    "Falconsai/nsfw_image_detection": [
        ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection"),
        AutoModelForImageClassification.from_pretrained(
            "Falconsai/nsfw_image_detection"
        ),
    ],
    "nateraw/vit-age-classifier": [
        ViTFeatureExtractor.from_pretrained("nateraw/vit-age-classifier"),
        ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier"),
    ],
    "microsoft/beit-base-patch16-224-pt22k-ft22k": [
        BeitImageProcessor.from_pretrained(
            "microsoft/beit-base-patch16-224-pt22k-ft22k"
        ),
        BeitForImageClassification.from_pretrained(
            "microsoft/beit-base-patch16-224-pt22k-ft22k"
        ),
    ],
    "microsoft/resnet-18": [
        AutoImageProcessor.from_pretrained("microsoft/resnet-18"),
        AutoModelForImageClassification.from_pretrained("microsoft/resnet-18"),
    ],
    "nvidia/mit-b0": [
        SegformerImageProcessor.from_pretrained("nvidia/mit-b0"),
        SegformerForImageClassification.from_pretrained("nvidia/mit-b0"),
    ],
    "google/vit-base-patch16-384": [
        ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-384"),
        ViTForImageClassification.from_pretrained("google/vit-base-patch16-384"),
    ],
    "facebook/convnext-large-224": [
        ConvNextImageProcessor.from_pretrained("facebook/convnext-large-224"),
        ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224"),
    ],
    "google/vit-large-patch16-224": [
        ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224"),
        ViTForImageClassification.from_pretrained("google/vit-large-patch16-224"),
    ],
    "facebook/deit-base-distilled-patch16-224": [
        AutoFeatureExtractor.from_pretrained(
            "facebook/deit-base-distilled-patch16-224"
        ),
        DeiTForImageClassificationWithTeacher.from_pretrained(
            "facebook/deit-base-distilled-patch16-224"
        ),
    ],
    "google/vit-base-patch32-384": [
        ViTFeatureExtractor.from_pretrained("google/vit-base-patch32-384"),
        ViTForImageClassification.from_pretrained("google/vit-base-patch32-384"),
    ],
    "microsoft/resnet-101": [
        AutoFeatureExtractor.from_pretrained("microsoft/resnet-101"),
        ResNetForImageClassification.from_pretrained("microsoft/resnet-101"),
    ],
    "microsoft/cvt-13": [
        AutoFeatureExtractor.from_pretrained("microsoft/cvt-13"),
        CvtForImageClassification.from_pretrained("microsoft/cvt-13"),
    ],
    "google/vit-large-patch16-384": [
        ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-384"),
        ViTForImageClassification.from_pretrained("google/vit-large-patch16-384"),
    ],
    "microsoft/beit-base-patch16-224": [
        BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224"),
        BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224"),
    ],
    "microsoft/beit-large-patch16-512": [
        BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-512"),
        BeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-512"),
    ],
    "microsoft/resnet-152": [
        AutoFeatureExtractor.from_pretrained("microsoft/resnet-152"),
        ResNetForImageClassification.from_pretrained("microsoft/resnet-152"),
    ],
    "google/mobilenet_v2_1.0_224": [
        AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224"),
        AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224"),
    ],
    "google/efficientnet-b7": [
        EfficientNetImageProcessor.from_pretrained("google/efficientnet-b7"),
        EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7"),
    ],
    "facebook/convnext-base-224": [
        ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224"),
        ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224"),
    ],
}


def test_image_classification_eval():
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    print("finish load dataset")

    for model_id in model_dict:
        # try:
        processor, model = image_classification_configutation[model_id]
        model = model.to(cuda_device)

        inputs = processor(image, return_tensors="pt", device=cuda_device)

        inputs["pixel_values"] = inputs["pixel_values"].to(cuda_device)
        if args.precision == "fp16":
            inputs["pixel_values"] = inputs["pixel_values"].to(
                device=cuda_device, dtype=torch.float16
            )
            model = model.to(cuda_device, dtype=torch.float16)
        elif args.precision == "bf16":
            inputs["pixel_values"] = inputs["pixel_values"].to(
                device=cuda_device, dtype=torch.bfloat16
            )
            model = model.to(cuda_device, dtype=torch.bfloat16)
        else:
            inputs["pixel_values"] = inputs["pixel_values"].to(cuda_device)
            model = model.to(cuda_device)
        if args.backend == "torch_compile":
            model = torch.compile(model)

        with torch.no_grad():
            logits = model(**inputs).logits

            # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        print("-" * 60)
        print(
            "Testing model name:{}, result device:{}".format(
                model_id, logits.device.type
            )
        )
        print(
            "Testing image classification model name {}, result:{}".format(
                model_id, model.config.id2label[predicted_label]
            )
        )
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_image_classification_eval()
