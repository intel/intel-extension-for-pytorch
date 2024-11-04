from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import numpy as np
import evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoModel
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

id2label = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
}
label2id = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12,
}

model_list = [
    "distilbert/distilbert-base-uncased",
    "dslim/bert-base-NER",
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
    "blaze999/Medical-NER",
    "ml6team/keyphrase-extraction-distilbert-inspec",
    "KoichiYasuoka/bert-base-thai-upos",
    "akdeniz27/bert-base-turkish-cased-ner",
    "51la5/roberta-large-NER",
    "Babelscape/wikineural-multilingual-ner",
    "dslim/bert-large-NER",
    "Davlan/bert-base-multilingual-cased-ner-hrl",
    "pierreguillou/ner-bert-base-cased-pt-lenerbr",
    "KB/bert-base-swedish-cased-ner",
    "NlpHUST/ner-vietnamese-electra-base",
    "Jean-Baptiste/camembert-ner",
    "SenswiseData/bert_cased_ner",
    "dslim/distilbert-NER",
    "ml6team/keyphrase-extraction-kbir-inspec",
    "ugaray96/biobert_ncbi_disease_ner",
    "Jean-Baptiste/roberta-large-ner-english",
    "NbAiLab/nb-bert-base-ner",
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
    "distilbert/distilbert-base-uncased": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "dslim/bert-base-NER": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "blaze999/Medical-NER": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "ml6team/keyphrase-extraction-distilbert-inspec": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "KoichiYasuoka/bert-base-thai-upos": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "akdeniz27/bert-base-turkish-cased-ner": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "51la5/roberta-large-NER": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "Babelscape/wikineural-multilingual-ner": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "dslim/bert-large-NER": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "Davlan/bert-base-multilingual-cased-ner-hrl": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "pierreguillou/ner-bert-base-cased-pt-lenerbr": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "KB/bert-base-swedish-cased-ner": [
        AutoTokenizer.from_pretrained,
        AutoModel.from_pretrained,
    ],
    "NlpHUST/ner-vietnamese-electra-base": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "Jean-Baptiste/camembert-ner": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "SenswiseData/bert_cased_ner": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "dslim/distilbert-NER": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "ml6team/keyphrase-extraction-kbir-inspec": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "ugaray96/biobert_ncbi_disease_ner": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "Jean-Baptiste/roberta-large-ner-english": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
    "NbAiLab/nb-bert-base-ner": [
        AutoTokenizer.from_pretrained,
        AutoModelForTokenClassification.from_pretrained,
    ],
}


wnut = load_dataset("wnut_17", trust_remote_code=True)
label_list = wnut["train"].features["ner_tags"].feature.names

example = wnut["train"][0]


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def test_trainer_one(model_id):

    tokenizer_type, model_type = model_configurations[model_id]
    tokenizer = tokenizer_type(model_id, trust_remote_code=True)
    tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
    seqeval = evaluate.load("seqeval")

    labels = [label_list[i] for i in example["ner_tags"]]

    model = model_type(
        model_id,
        num_labels=13,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        report_to="none",
        output_dir=f"./token-classification-{model_id}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="no",
        load_best_model_at_end=False,
        push_to_hub=False,
        save_safetensors=False,
        fp16=fp16_enable,
        bf16=bf16_enable,
        torch_compile=torch_compile,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_wnut["train"],
        eval_dataset=tokenized_wnut["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("*" * 60)
    print(f"Test training model:{model_id} successfully")


def test_token_classification_train():
    for model_id in model_list:
        try:
            test_trainer_one(model_id)
            print(f"Testing model success: {model_id}")
        except Exception as e:
            print(f"Testing model fail: {model_id}")
            print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_token_classification_train()
