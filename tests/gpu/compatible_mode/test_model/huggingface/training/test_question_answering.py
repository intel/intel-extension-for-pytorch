from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DistilBertForQuestionAnswering,
)
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    PreTrainedTokenizerFast,
    GemmaForCausalLM,
)
from transformers import AutoModelForSeq2SeqLM
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

squad = load_dataset("squad", split="train[:5000]")

squad = squad.train_test_split(test_size=0.2)

# dataset organized as follow
# {'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
#  'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',  # noqa:B950
#  'id': '5733be284776f41900661182',
#  'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
#  'title': 'University_of_Notre_Dame'
# }


model_list = [
    "distilbert/distilbert-base-uncased",
    "deepset/roberta-base-squad2",
    "deepset/bert-large-uncased-whole-word-masking-squad2",
    "distilbert/distilbert-base-cased-distilled-squad",
    "deepset/roberta-large-squad2",
    "sjrhuschlee/flan-t5-base-squad2",
    "deepset/minilm-uncased-squad2",
    "deepset/tinyroberta-squad2",
    "distilbert/distilbert-base-uncased-distilled-squad",
    "uer/roberta-base-chinese-extractive-qa",
    "deepset/deberta-v3-large-squad2",
    "deepset/deberta-v3-base-squad2",
    "deepset/xlm-roberta-large-squad2",
    "pedramyazdipoor/persian_xlm_roberta_large",
    "DeepMount00/Gemma_QA_ITA_v3",
    "VietAI/vit5-base",
    "deepset/xlm-roberta-base-squad2-distilled",
    "pierreguillou/bert-base-cased-squad-v1.1-portuguese",
    "deepset/xlm-roberta-base-squad2",
    "VietAI/vit5-large",
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
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "deepset/roberta-base-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "deepset/bert-large-uncased-whole-word-masking-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "distilbert/distilbert-base-cased-distilled-squad": [
        PreTrainedTokenizerFast.from_pretrained,
        DistilBertModel.from_pretrained,
    ],
    "deepset/roberta-large-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "sjrhuschlee/flan-t5-base-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "deepset/minilm-uncased-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "deepset/tinyroberta-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "distilbert/distilbert-base-uncased-distilled-squad": [
        DistilBertTokenizer.from_pretrained,
        DistilBertForQuestionAnswering.from_pretrained,
    ],
    "uer/roberta-base-chinese-extractive-qa": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "deepset/deberta-v3-large-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "deepset/deberta-v3-base-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "deepset/xlm-roberta-large-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "pedramyazdipoor/persian_xlm_roberta_large": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "DeepMount00/Gemma_QA_ITA_v3": [
        AutoTokenizer.from_pretrained,
        GemmaForCausalLM.from_pretrained,
    ],
    "VietAI/vit5-base": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
    "deepset/xlm-roberta-base-squad2-distilled": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "pierreguillou/bert-base-cased-squad-v1.1-portuguese": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "deepset/xlm-roberta-base-squad2": [
        AutoTokenizer.from_pretrained,
        AutoModelForQuestionAnswering.from_pretrained,
    ],
    "VietAI/vit5-large": [
        AutoTokenizer.from_pretrained,
        AutoModelForSeq2SeqLM.from_pretrained,
    ],
}


def test_trainer_one(model_id):
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenizer_type, model_type = model_configurations[model_id]

    tokenizer = tokenizer_type(model_id, trust_remote_code=True)
    tokenized_squad = squad.map(
        preprocess_function, batched=True, remove_columns=squad["train"].column_names
    )

    data_collator = DefaultDataCollator()

    model = model_type(model_id, ignore_mismatched_sizes=True, trust_remote_code=True)

    training_args = TrainingArguments(
        report_to="none",
        output_dir=f"./question-answering-{model_id}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        save_strategy="no",
        fsdp=False,
        load_best_model_at_end=False,
        fp16=fp16_enable,
        bf16=bf16_enable,
        torch_compile=torch_compile,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


def test_question_answering_train():
    for model_id in model_list:
        # try:
        test_trainer_one(model_id)
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_question_answering_train()
