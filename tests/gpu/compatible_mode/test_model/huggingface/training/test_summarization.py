from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)

# dataset organized as below
# {'summary', 'text', 'title'}


model_list = [
    "google-t5/t5-small",
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6",
    "philschmid/bart-large-cnn-samsum",
    "google/pegasus-xsum",
    "Falconsai/medical_summarization",
    "transformer3/H2-keywordextractor",
    "sshleifer/distilbart-cnn-12-3",
    "google/pegasus-multi_news",
    "nandakishormpai/t5-small-machine-articles-tag-generation",
    "slauw87/bart_summarisation",
    "cnicu/t5-small-booksum",
    "knkarthick/MEETING_SUMMARY",
    "jotamunz/billsum_tiny_summarization",
    "sshleifer/distilbart-cnn-6-6",
    "human-centered-summarization/financial-summarization-pegasus",
    "google/pegasus-cnn_dailymail",
    "IlyaGusev/mbart_ru_sum_gazeta",
    "lidiya/bart-large-xsum-samsum",
    "gsarti/it5-base-news-summarization",
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
    "google-t5/t5-small": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "sshleifer/distilbart-cnn-12-6": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "philschmid/bart-large-cnn-samsum": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "google/pegasus-xsum": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "Falconsai/medical_summarization": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "Falconsai/text_summarization": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "facebook/bart-large-cnn": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "transformer3/H2-keywordextractor": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "sshleifer/distilbart-cnn-12-3": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "google/pegasus-multi_news": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "nandakishormpai/t5-small-machine-articles-tag-generation": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "slauw87/bart_summarisation": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "cnicu/t5-small-booksum": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "knkarthick/MEETING_SUMMARY": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "jotamunz/billsum_tiny_summarization": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "sshleifer/distilbart-cnn-6-6": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "human-centered-summarization/financial-summarization-pegasus": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "google/pegasus-cnn_dailymail": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "IlyaGusev/mbart_ru_sum_gazeta": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "lidiya/bart-large-xsum-samsum": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "gsarti/it5-base-news-summarization": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
}


def test_trainer_one(model_id):
    tokenizer_type, model_type = model_configurations[model_id]

    tokenizer = tokenizer_type(model_id, trust_remote_code=True)

    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(
            text_target=examples["summary"], max_length=128, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_billsum = billsum.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_id)
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    model = model_type(model_id, ignore_mismatched_sizes=True, trust_remote_code=True)

    training_args = Seq2SeqTrainingArguments(
        report_to="none",
        output_dir=f"./summarization-{model_id}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=False,
        fp16=True,
        push_to_hub=False,
        save_strategy="no",
        fsdp=False,
        load_best_model_at_end=False,
        fp16=fp16_enable,
        bf16=bf16_enable,
        torch_compile=torch_compile,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_billsum["train"],
        eval_dataset=tokenized_billsum["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("*" * 60)
    print(f"Test training model:{model_id} successfully")


def test_summarization_train():
    for model_id in model_list:
        # try:
        test_trainer_one(model_id)
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_summarization_train()
