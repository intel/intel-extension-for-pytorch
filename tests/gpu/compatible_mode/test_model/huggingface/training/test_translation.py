# encoding: UTF-8
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import numpy as np
import evaluate
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

# dataset are organized as follow
# {'id': '90560',
#  'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
#   'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}


model_list = [
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "Helsinki-NLP/opus-mt-mul-en",
    "facebook/nllb-200-distilled-600M",
    "Helsinki-NLP/opus-mt-de-en",
    "google-t5/t5-3b",
    "optimum/t5-small",
    "facebook/wmt19-ru-en",
    "Helsinki-NLP/opus-mt-fr-de",
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
    "optimum/t5-small": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "en",
        "fr",
        "translate English to French: ",
    ],
    "google-t5/t5-small": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "en",
        "fr",
        "translate English to French: ",
    ],
    "google-t5/t5-base": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "en",
        "fr",
        "translate English to French: ",
    ],
    "google-t5/t5-large": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "en",
        "fr",
        "translate English to French: ",
    ],
    "Helsinki-NLP/opus-mt-mul-en": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "fr",
        "en",
        "translate French to English: ",
    ],
    "facebook/nllb-200-distilled-600M": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "en",
        "fr",
        "translate English to French: ",
    ],
    "Helsinki-NLP/opus-mt-de-en": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "de",
        "en",
        "translate de to English: ",
    ],
    "google-t5/t5-3b": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "en",
        "fr",
        "translate English to French: ",
    ],
    "facebook/wmt19-ru-en": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "ru",
        "en",
        "translate ru to English: ",
    ],
    "Helsinki-NLP/opus-mt-fr-de": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
        "fr",
        "de",
        "translate fr to de: ",
    ],
}


def test_trainer_one(model_id):
    tokenizer_type, model_type, s_lang, t_lang, pre_fix = model_configurations[model_id]

    books = load_dataset("opus_books", f"{s_lang}-{t_lang}")

    books = books["train"].train_test_split(test_size=0.2)

    tokenizer = tokenizer_type(model_id, trust_remote_code=True)
    source_lang = s_lang
    target_lang = t_lang
    prefix = pre_fix

    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=128, truncation=True
        )
        return model_inputs

    tokenized_books = books.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    model = mode_type(checkpoint, trust_remote_code=True)

    training_args = Seq2SeqTrainingArguments(
        report_to="none",
        output_dir=f"./translation-{model_id}",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=False,
        save_safetensors=False,
        push_to_hub=False,
        fsdp=False,
        fp16=fp16_enable,
        bf16=bf16_enable,
        torch_compile=torch_compile,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("*" * 60)
    print(f"Test training model:{model_id} successfully")


def test_translation_train():
    for model_id in model_list:
        try:
            test_trainer_one(model_id)
            print(f"Testing model success: {model_id}")
        except Exception as e:
            print(f"Testing model fail: {model_id}")
            print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_translation_train()
