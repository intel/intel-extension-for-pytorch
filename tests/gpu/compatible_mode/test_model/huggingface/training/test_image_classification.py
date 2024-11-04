from datasets import load_dataset
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import DefaultDataCollator
import evaluate
import numpy as np
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import ResNetForImageClassification, ConvNextImageProcessor
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
from transformers import SegformerImageProcessor, SegformerForImageClassification
from transformers import (
    AutoFeatureExtractor,
    SwinForImageClassification,
    ConvNextForImageClassification,
)
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()


class dataset:
    def __init__(self, name, split, test_size):
        dataset = load_dataset(name, split=split)
        self.dataset = dataset.train_test_split(test_size=test_size)

        labels = self.dataset["train"].features["label"].names

        label2id, id2label = dict(), dict()

        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        self.labels = labels
        self.label2id = label2id
        self.id2label = id2label


model_list = [
    "google/vit-base-patch16-224-in21k",
    "microsoft/resnet-50",
    "google/vit-base-patch16-224",
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "Falconsai/nsfw_image_detection",
    "AdamCodd/vit-base-nsfw-detector",
    "microsoft/swinv2-tiny-patch4-window16-256",
    "google/mobilenet_v1_0.75_192",
    "google/vit-large-patch32-384",
    "facebook/convnextv2-tiny-1k-224",
    "microsoft/dit-base-finetuned-rvlcdip",
    "microsoft/resnet-18",
    "google/mobilenet_v2_1.0_224",
    "nvidia/mit-b0",
    "google/vit-large-patch16-224",
    "microsoft/swin-base-patch4-window12-384",
    "facebook/convnext-base-224-22k",
    "nvidia/mit-b5",
    "microsoft/swin-tiny-patch4-window7-224",
    "nvidia/mit-b1",
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
    model_list = [args.model]
torch_compile = False
if args.backend == "torch_compile":
    torch_compile = True
fp16_enable = False
bf16_enable = False
if args.precision == "fp16":
    fp16_enable = True
elif args.precision == "bf16":
    bf16_enable = True

model_configurations = {
    "google/vit-base-patch16-224-in21k": [
        AutoImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "microsoft/resnet-50": [
        AutoImageProcessor.from_pretrained,
        ResNetForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "google/vit-base-patch16-224": [
        ViTImageProcessor.from_pretrained,
        ViTForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "microsoft/beit-base-patch16-224-pt22k-ft22k": [
        BeitImageProcessor.from_pretrained,
        BeitForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "Falconsai/nsfw_image_detection": [
        ViTImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "AdamCodd/vit-base-nsfw-detector": [
        ViTImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "microsoft/swinv2-tiny-patch4-window16-256": [
        AutoImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "google/mobilenet_v1_0.75_192": [
        AutoImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "google/vit-large-patch32-384": [
        ViTImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "facebook/convnextv2-tiny-1k-224": [
        AutoImageProcessor.from_pretrained,
        ConvNextV2ForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "microsoft/dit-base-finetuned-rvlcdip": [
        AutoImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "microsoft/resnet-18": [
        AutoImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "google/mobilenet_v2_1.0_224": [
        AutoImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "nvidia/mit-b0": [
        SegformerImageProcessor.from_pretrained,
        SegformerForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "google/vit-large-patch16-224": [
        ViTImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "microsoft/swin-base-patch4-window12-384": [
        AutoFeatureExtractor.from_pretrained,
        SwinForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "facebook/convnext-base-224-22k": [
        ConvNextImageProcessor.from_pretrained,
        ConvNextForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "nvidia/mit-b5": [
        SegformerImageProcessor.from_pretrained,
        SegformerForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "microsoft/swin-tiny-patch4-window7-224": [
        AutoImageProcessor.from_pretrained,
        AutoModelForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
    "nvidia/mit-b1": [
        SegformerImageProcessor.from_pretrained,
        SegformerForImageClassification.from_pretrained,
        dataset("food101", "train[:5000]", 0.2),
    ],
}


def test_trainer_one(model_id):
    image_processor_type, model_type, dataset = model_configurations[model_id]

    image_processor = image_processor_type(model_id, trust_remote_code=True)

    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [
            _transforms(img.convert("RGB")) for img in examples["image"]
        ]
        del examples["image"]
        return examples

    dataset.dataset = dataset.dataset.with_transform(transforms)
    data_collator = DefaultDataCollator()
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = model_type(
        model_id,
        num_labels=len(dataset.labels),
        id2label=dataset.id2label,
        label2id=dataset.label2id,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    print("Testing model type:", model)

    training_args = TrainingArguments(
        report_to="none",
        output_dir=f"./image-classification_{model_id}",
        remove_unused_columns=False,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        save_strategy="no",
        fsdp=False,
        fp16=fp16_enable,
        bf16=bf16_enable,
        torch_compile=torch_compile,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset.dataset["train"],
        eval_dataset=dataset.dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("*" * 60)
    print(f"Test training model:{model_id} successfully")


def test_image_classification_train():
    for model_id in model_list:
        # try:
        test_trainer_one(model_id)
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        # print(f"Testing model fail: {model_id}")
        # print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_image_classification_train()
