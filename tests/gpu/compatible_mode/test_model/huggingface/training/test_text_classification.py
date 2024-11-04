from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
from transformers import DataCollatorWithPadding
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import (
    DistilBertTokenizer,
    AutoModel,
)

import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

imdb = load_dataset("imdb")
# dataset are organized as below
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 25000
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 25000
#     })
#     unsupervised: Dataset({
#         features: ['text', 'label'],
#         num_rows: 50000
#     })
# })

model_list = [
    "distilbert/distilbert-base-uncased",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "avichr/heBERT_sentiment_analysis",
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "papluca/xlm-roberta-base-language-detection"
    "cardiffnlp/twitter-roberta-base-sentiment",
    "Ashishkr/query_wellformedness_score",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "martin-ha/toxic-comment-model",
    "BAAI/bge-reranker-v2-m3",
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


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
num_labels = 2

# config tokenizer and
model_configurations = {
    "distilbert/distilbert-base-uncased": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english": [
        DistilBertTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "cardiffnlp/twitter-roberta-base-sentiment-latest": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "avichr/heBERT_sentiment_analysis": [
        AutoTokenizer.from_pretrained,
        AutoModel.from_pretrained,
    ],
    "cross-encoder/ms-marco-TinyBERT-L-2-v2": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "papluca/xlm-roberta-base-language-detection": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "cardiffnlp/twitter-roberta-base-sentiment": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "Ashishkr/query_wellformedness_score": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "cross-encoder/ms-marco-MiniLM-L-6-v2": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "cross-encoder/ms-marco-MiniLM-L-12-v2": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "martin-ha/toxic-comment-model": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
    "BAAI/bge-reranker-v2-m3": [
        AutoTokenizer.from_pretrained,
        AutoModelForSequenceClassification.from_pretrained,
    ],
}


def test_trainer_one(model_id):

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenizer_type, model_type = model_configurations[model_id]

    tokenizer = tokenizer_type(
        model_id, trust_remote_code=True, ignore_mismatched_sizes=True
    )

    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = model_type(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    accuracy = evaluate.load("accuracy")

    training_args = TrainingArguments(
        report_to="none",
        output_dir=f"./text-classification_{model_id}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="no",
        load_best_model_at_end=True,
        save_safetensors=False,
        push_to_hub=False,
        fsdp=False,
        fp16=fp16_enable,
        bf16=bf16_enable,
        torch_compile=torch_compile,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("*" * 60)
    print(f"Test training model:{model_id} successfully")


def test_text_classification_train():
    for model_id in model_list:
        # try:
        test_trainer_one(model_id)
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_text_classification_train()
