import torch
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

import transformers

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from transformers import BlipProcessor, BlipForConditionalGeneration


# from PIL import Image
# import requests


cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model_dict = [
    # "Text-classification": {
    "ProsusAI/finbert",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "SamLowe/roberta-base-go_emotions",
    "j-hartmann/emotion-english-distilroberta-base",
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "cardiffnlp/twitter-roberta-base-sentiment",
    "papluca/xlm-roberta-base-language-detection",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "vectara/hallucination_evaluation_model",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "yiyanghkust/finbert-tone",
    "FacebookAI/roberta-large-mnli",
    "facebook/fasttext-language-identification",  # Fail due to repo issue
    "unitary/toxic-bert",
    "finiteautomata/bertweet-base-sentiment-analysis",
    "BAAI/bge-reranker-base",
    "notdiamond/notdiamond-0001",
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


def test_one(model_id):
    pipeline = transformers.pipeline(
        "sentiment-analysis", model=model_id, device=cuda_device, trust_remote_code=True
    )
    if args.precision == "fp16":
        pipeline.model = pipeline.model.to(torch.float16)
    elif args.precision == "bf16":
        pipeline.model = pipeline.model.to(torch.bfloat16)
    if args.backend == "torch_compile":
        model = torch.compile(pipeline.model)
        pipeline.model = model
    result = pipeline("Hello,world")[0]
    print("Text classification model name {}, result: {}".format(model_id, result))


def test_text_classification_eval():
    for model_id in model_dict:
        # try:
        test_one(model_id)

        # except Exception as e:
        #    print(f"error model name:{model_id}")
        #    print("Exception:", e)


# image_to_text_configuration = {
#    "microsoft/trocr-base-handwritten": [
#        "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg",
#        TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten"),
#        VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten"),
#    ],
#    "nlpconnect/vit-gpt2-image-captioning": [
#        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
#        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"),
#        BlipForConditionalGeneration.from_pretrained(
#            "Salesforce/blip-image-captioning-large"
#        ),
#    ],
#    "Salesforce/blip-image-captioning-base": [
#        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
#        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"),
#        BlipForConditionalGeneration.from_pretrained(
#            "Salesforce/blip-image-captioning-large"
#        ),
#    ],
# }


# def test_Image_To_Text_eval():
#    for model_id in model_dict["Image-To-Text"]:
#        try:
#            url, processor, model = image_to_text_configuration[model_id]
#            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#
#            pixel_values = processor(
#                images=image, return_tensors="pt", device=cuda_device
#            ).pixel_values
#            pixel_values = pixel_values.to(cuda_device)
#            model = model.to(cuda_device)

#            generated_ids = model.generate(pixel_values)
#            generated_text = processor.batch_decode(
#                generated_ids, skip_special_tokens=True
#            )[0]
#            print("-" * 60)
#            print("Testing model:{}, result:{}".format(model_id, generated_text))
#            print(f"Testing model success: {model_id}")
#        except Exception as e:
#            print(f"Testing model fail: {model_id}")
#            print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_text_classification_eval()
