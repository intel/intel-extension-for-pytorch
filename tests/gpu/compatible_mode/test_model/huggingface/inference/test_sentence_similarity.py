# TODO: tested but all failed
import torch
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model_dict = [
    # "Sentence_Similarity": {
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/bert-base-nli-mean-tokens",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/msmarco-bert-base-dot-v5",
    "Alibaba-NLP/gte-large-en-v1.5",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
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


text_to_image_configuration = {
    "sentence-transformers/all-MiniLM-L6-v2": [
        "This is an example sentence",
        "Each sentence is converted",
    ],
    "sentence-transformers/all-MiniLM-L12-v2": [
        "This is an example sentence",
        "Each sentence is converted",
    ],
    "sentence-transformers/all-mpnet-base-v2": [
        "This is an example sentence",
        "Each sentence is converted",
    ],
    "sentence-transformers/paraphrase-MiniLM-L6-v2": [
        "This is an example sentence",
        "Each sentence is converted",
    ],
    "sentence-transformers/bert-base-nli-mean-tokens": [
        "This is an example sentence",
        "Each sentence is converted",
    ],
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": [
        "This is an example sentence",
        "Each sentence is converted",
    ],
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": [
        "This is an example sentence",
        "Each sentence is converted",
    ],
    "sentence-transformers/msmarco-bert-base-dot-v5": [
        "This is an example sentence",
        "Each sentence is converted",
    ],
    "Alibaba-NLP/gte-large-en-v1.5": [
        "That is a happy person",
        "That is a very happy person",
    ],
    "sentence-transformers/distiluse-base-multilingual-cased-v2": [
        "That is a happy person",
        "That is a very happy person",
    ],
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": [
        "That is a happy person",
        "That is a very happy person",
    ],
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": [
        "That is a happy person",
        "That is a very happy person",
    ],
    "sentence-transformers/all-roberta-large-v1": [
        "That is a happy person",
        "That is a very happy person",
    ],
}


def test_sentence_similarity_eval():
    for model_id in model_dict:
        if args.precision == "fp16":
            model = SentenceTransformer(
                model_id,
                device=cuda_device,
                trust_remote_code=True,
                dtype=torch.float16,
            )
        elif args.precision == "bf16":
            model = SentenceTransformer(
                model_id,
                device=cuda_device,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            )
        else:
            model = SentenceTransformer(
                model_id, device=cuda_device, trust_remote_code=True
            )
        if args.backend == "torch_compile":
            model = torch.compile(model)
        sentences = text_to_image_configuration[model_id]
        embeddings = model.encode(sentences)

        print("*" * 60)
        print(
            f"Testing model:{model_id}, result:{cos_sim(embeddings[0], embeddings[1])}"
        )
        print("result device:", model.device)
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        # print(f"Testing model fail: {model_id}")
        # print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_sentence_similarity_eval()
