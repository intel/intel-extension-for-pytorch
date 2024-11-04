# encoding: UTF-8
import torch
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

from transformers import pipeline


cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model_dict = [
    # "Text_Zero_Shot_Pipeline": {
    "tasksource/deberta-small-long-nli",
    "facebook/bart-large-mnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "cross-encoder/nli-roberta-base",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-base",
    "joeddav/xlm-roberta-large-xnli",
    "cross-encoder/nli-distilroberta-base",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "sileod/deberta-v3-base-tasksource-nli",
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

text_zero_shot_configuration = {
    "tasksource/deberta-small-long-nli": [
        "one day I will see the world",
        ["travel", "cooking", "dancing"],
    ],
    "facebook/bart-large-mnli": [
        "one day I will see the world",
        ["travel", "cooking", "dancing"],
    ],
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": [
        "Angela Merkel is a politician in Germany and leader of the CDU",
        ["politics", "economy", "entertainment", "environment"],
    ],
    "cross-encoder/nli-roberta-base": [
        "Apple just announced the newest iPhone X",
        ["technology", "sports", "politics"],
    ],
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0": [
        "Angela Merkel is a politician in Germany and leader of the CDU",
        ["politics", "economy", "entertainment", "environment"],
    ],
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": [
        "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU",
        ["politics", "economy", "entertainment", "environment"],
    ],
    "cross-encoder/nli-deberta-v3-base": [
        "A man is eating pizza",
        ["contradiction", "entailment", "neutral"],
    ],
    "cross-encoder/nli-deberta-base": [
        "Apple just announced the newest iPhone X",
        ["technology", "sports", "politics"],
    ],
    "joeddav/xlm-roberta-large-xnli": [
        "За кого вы голосуете в 2020 году?",
        ["Europe", "public health", "politics"],
    ],
    "cross-encoder/nli-distilroberta-base": [
        "A man is eating pizza",
        ["contradiction", "entailment", "neutral"],
    ],
    "cross-encoder/nli-MiniLM2-L6-H768": [
        "A man is eating pizza",
        ["contradiction", "entailment", "neutral"],
    ],
    "sileod/deberta-v3-base-tasksource-nli": [
        "one day I will see the world",
        ["travel", "cooking", "dancing"],
    ],
}


def test_one(model_id):
    classifier = pipeline(
        "zero-shot-classification", model=model_id, device=cuda_device
    )
    if args.precision == "fp16":
        classifier.model = classifier.model.to(torch.float16)
    elif args.precision == "bf16":
        classifier.model = classifier.model.to(torch.bfloat16)
    if args.backend == "torch_compile":
        model = torch.compile(classifier.model)
        classifier.model = model
    text, candidate_labels = text_zero_shot_configuration[model_id]
    res = classifier(text, candidate_labels)

    print("*" * 60)
    print("Testing model:", model_id)
    print("result:", res)
    print("result device:", classifier.device)


def test_text_zero_shot_pipeline():
    for model_id in model_dict:
        # try:
        test_one(model_id)
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_text_zero_shot_pipeline()
