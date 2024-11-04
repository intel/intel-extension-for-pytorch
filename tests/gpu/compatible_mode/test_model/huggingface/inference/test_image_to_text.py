import torch
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

from PIL import Image
import requests

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition


cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model_dict = [
    # "Image-To-Text": [
    "microsoft/trocr-base-handwritten",  # Failed for tools reason
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip2-opt-2.7b",  # Failed for tools reason but have fix didn't land
    "microsoft/trocr-small-handwritten",
    "facebook/nougat-base",
    "Salesforce/instructblip-vicuna-7b",
    "microsoft/trocr-large-printed",
    "Salesforce/blip2-flan-t5-xl",
    "microsoft/trocr-base-printed",
    "alibaba-damo/mgp-str-base",
    # ],
    # "Image-To-Text-Pipeline": [
    "nlpconnect/vit-gpt2-image-captioning",
    # ],
]
Image_To_Text_Pipeline = ["nlpconnect/vit-gpt2-image-captioning"]

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

image_to_text_configuration = {
    "microsoft/trocr-base-handwritten": [
        "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg",
        TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten"),
        VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten"),
    ],
    "nlpconnect/vit-gpt2-image-captioning": [
        "https://ankur3107.github.io/assets/images/image-captioning-example.png"
    ],
    "llava-hf/llava-1.5-7b-hf": [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
    ],
    "Salesforce/blip-image-captioning-large": [
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"),
        BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ),
    ],
    "Salesforce/blip-image-captioning-base": [
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
        BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ),
    ],
    "Salesforce/blip2-opt-2.7b": [
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
        Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b"),
        Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", device_map="auto"
        ),
    ],
    "microsoft/trocr-small-handwritten": [
        "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg",
        TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten"),
        VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten"),
    ],
    "Salesforce/instructblip-vicuna-7b": [
        "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg",
        InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b"),
        InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b"
        ),
    ],
    "microsoft/trocr-large-printed": [
        "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg",
        TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed"),
        VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed"),
    ],
    "Salesforce/blip2-flan-t5-xl": [
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
        Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl"),
        Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", device_map="auto"
        ),
    ],
    "microsoft/trocr-base-printed": [
        "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg",
        TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed"),
        VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed"),
    ],
    "alibaba-damo/mgp-str-base": [
        "https://i.postimg.cc/ZKwLg2Gw/367-14.png",
        MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base"),
        MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base"),
    ],
}


def test_one(model_id):
    url, processor, model = image_to_text_configuration[model_id]

    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    if args.precision == "fp16":
        model = model.to(device=cuda_device, dtype=torch.float16)
    elif args.precision == "bf16":
        model = model.to(device=cuda_device, dtype=torch.bfloat16)
    else:
        model = model.to(cuda_device)
    if args.backend == "torch_compile":
        model = torch.compile(model)

    pixel_values = processor(
        images=image, return_tensors="pt", device=cuda_device
    ).pixel_values
    # if args.precision == "fp16":
    #    pixel_values = processor(
    #        images=image, return_tensors="pt", device=cuda_device, dtype=torch.float16
    #    ).pixel_values
    #    model = model.to(cuda_device, dtype=torch.float16)
    # elif args.precision == "bf16":
    #    pixel_values = processor(
    #        images=image, return_tensors="pt", device=cuda_device, dtype=torch.bfloat16
    #    ).pixel_values
    #    model = model.to(cuda_device, dtype=torch.bfloat16)
    # else:
    #    pixel_values = processor(
    #        images=image, return_tensors="pt", device=cuda_device
    #    ).pixel_values
    #    model = model.to(cuda_device)
    # if args.backend == "torch_compile":
    #    model = torch.compile(model)
    generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("-" * 60)
    print(
        "Testing model:{} on device:{}, result:{}".format(
            model_id, generated_ids.device, generated_text
        )
    )


def test_Image_To_Text_eval():
    for model_id in model_dict:
        if model_id not in Image_To_Text_Pipeline:
            test_one(model_id)
            print(f"Testing model success: {model_id}")
        else:
            print("skip test_Image_To_Text_eval")
        # try:
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


def test_image_to_text_pipeline():
    for model_id in model_dict:
        if model_id in Image_To_Text_Pipeline:
            image_to_text = pipeline(
                "image-to-text", model=model_id, device=cuda_device
            )
            res = image_to_text(image_to_text_configuration[model_id])

            print("-" * 60)
            print("Testing model:{}, result:{}".format(model_id, res))


if __name__ == "__main__":
    test_Image_To_Text_eval()
    test_image_to_text_pipeline()
