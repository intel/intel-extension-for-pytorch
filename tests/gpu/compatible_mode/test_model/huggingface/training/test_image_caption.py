from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import NougatProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from transformers import AutoModel, AutoProcessor, MgpstrForSceneTextRecognition
from transformers import AutoModelForVision2Seq, MgpstrProcessor

from evaluate import load
from transformers import TrainingArguments, Trainer

import intel_extension_for_pytorch as ipex

ipex.compatible_mode()
# import wandb
# import os


model_list = [
    "microsoft/git-base",
    "nlpconnect/vit-gpt2-image-captioning",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip2-opt-2.7b",
    "facebook/nougat-base",
    "google/pix2struct-large",
    "Salesforce/blip2-flan-t5-xl",
    "Salesforce/blip2-opt-2.7b-coco",
    "google/pix2struct-textcaps-base",
    "microsoft/kosmos-2-patch14-224",
    "alibaba-damo/mgp-str-base",
    "unum-cloud/uform-gen2-qwen-500m",
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
    "microsoft/git-base": [
        AutoProcessor.from_pretrained,
        AutoModelForCausalLM.from_pretrained,
    ],
    "nlpconnect/vit-gpt2-image-captioning": [
        ViTImageProcessor.from_pretrained,
        VisionEncoderDecoderModel.from_pretrained,
    ],
    "Salesforce/blip-image-captioning-large": [
        BlipProcessor.from_pretrained,
        BlipForConditionalGeneration.from_pretrained,
    ],
    "Salesforce/blip-image-captioning-base": [
        BlipProcessor.from_pretrained,
        BlipForConditionalGeneration.from_pretrained,
    ],
    "Salesforce/blip2-opt-2.7b": [
        Blip2Processor.from_pretrained,
        Blip2ForConditionalGeneration.from_pretrained,
    ],
    "facebook/nougat-base": [
        NougatProcessor.from_pretrained,
        VisionEncoderDecoderModel.from_pretrained,
    ],
    "Salesforce/instructblip-vicuna-7b": [
        InstructBlipProcessor.from_pretrained,
        InstructBlipForConditionalGeneration.from_pretrained,
    ],
    "google/pix2struct-large": [
        Pix2StructProcessor.from_pretrained,
        Pix2StructForConditionalGeneration.from_pretrained,
    ],
    "Salesforce/blip2-flan-t5-xl": [
        Blip2Processor.from_pretrained,
        Blip2ForConditionalGeneration.from_pretrained,
    ],
    "Salesforce/blip2-opt-2.7b-coco": [
        Blip2Processor.from_pretrained,
        Blip2ForConditionalGeneration.from_pretrained,
    ],
    "google/pix2struct-textcaps-base": [
        Pix2StructProcessor.from_pretrained,
        Pix2StructForConditionalGeneration.from_pretrained,
    ],
    "microsoft/kosmos-2-patch14-224": [
        AutoProcessor.from_pretrained,
        AutoModelForVision2Seq.from_pretrained,
    ],
    "alibaba-damo/mgp-str-base": [
        MgpstrProcessor.from_pretrained,
        MgpstrForSceneTextRecognition.from_pretrained,
    ],
    "unum-cloud/uform-gen2-qwen-500m": [
        AutoProcessor.from_pretrained,
        AutoModel.from_pretrained,
    ],
}


def test_trainer_one(model_id):
    ds = load_dataset("svjack/pokemon-blip-captions-en-zh")
    # dataset are organized as below
    # DatasetDict({
    #     train: Dataset({
    #         features: ['image', 'en_text', 'zh_text'],
    #         num_rows: 833
    #     })
    # })

    ds = ds["train"].train_test_split(test_size=0.1)
    train_ds = ds["train"]
    test_ds = ds["test"]

    processor_type, model_type = model_configurations[model_id]
    processor = processor_type(model_id, trust_remote_code=True)

    def transforms(example_batch):
        images = [x for x in example_batch["image"]]  # noqa:C416
        captions = [x for x in example_batch["en_text"]]  # noqa:C416
        inputs = processor(images=images, text=captions, padding="max_length")
        # inputs.update({"labels": inputs["input_ids"]})
        if "input_ids" in inputs.keys():
            inputs.update({"labels": inputs["input_ids"]})
        elif "decoder_input_ids" in inputs.keys():
            inputs.update({"labels": inputs["decoder_input_ids"]})
        else:
            print(f"no ids in {inputs.keys()}")
            exit(-1)
        return inputs

    train_ds.set_transform(transforms)
    test_ds.set_transform(transforms)

    model = model_type(model_id, ignore_mismatched_sizes=True, trust_remote_code=True)

    wer = load("wer")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predicted = logits.argmax(-1)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = processor.batch_decode(
            predicted, skip_special_tokens=True
        )
        wer_score = wer.compute(
            predictions=decoded_predictions, references=decoded_labels
        )
        return {"wer_score": wer_score}

    training_args = TrainingArguments(
        report_to="none",
        output_dir=f"./image-caption-{model_id}",
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
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("*" * 60)
    print(f"Test training model:{model_id} successfully")


def test_image_caption_train():
    for model_id in model_list:
        # try:
        test_trainer_one(model_id)
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_image_caption_train()
