import torch
import intel_extension_for_pytorch as ipex
ipex.compatible_mode()

import transformers
from transformers import AutoImageProcessor, ResNetForImageClassification, BeitFeatureExtractor, EfficientNetImageProcessor
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoModelForImageClassification
from transformers import ViTFeatureExtractor
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import SegformerImageProcessor, SegformerForImageClassification
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
from transformers import CvtForImageClassification, EfficientNetForImageClassification
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import YolosImageProcessor, YolosForObjectDetection, YolosFeatureExtractor
from transformers import DeformableDetrForObjectDetection, ConditionalDetrForObjectDetection
from transformers import Mask2FormerForUniversalSegmentation
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BlipProcessor, BlipForConditionalGeneration



from PIL import Image
import requests



from datasets import load_dataset

cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model_dict = {
    "Text_Generation": {
        "meta-llama/Meta-Llama-3-8B",
        # "mistralai/Mixtral-8x7B-Instruct-v0.1",

    },
    "Text-classification": {
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
        "facebook/fasttext-language-identification",
        "unitary/toxic-bert",
        "finiteautomata/bertweet-base-sentiment-analysis",
        "BAAI/bge-reranker-base",
        "notdiamond/notdiamond-0001",
    },
    "Image-Classification": {
        "microsoft/resnet-50",
        "google/vit-base-patch16-224",
        "Falconsai/nsfw_image_detection",
        "nateraw/vit-age-classifier",
        "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "microsoft/resnet-18",
        "nvidia/mit-b0",
        "google/vit-base-patch16-384",
        "google/vit-large-patch16-224",
        "facebook/deit-base-distilled-patch16-224",
        "google/vit-base-patch32-384",
        "microsoft/resnet-101",
        "microsoft/cvt-13",
        "google/vit-large-patch16-384",
        "microsoft/beit-base-patch16-224",
        "microsoft/beit-large-patch16-512",
        "microsoft/resnet-152",
        "google/mobilenet_v2_1.0_224",
        "google/efficientnet-b7",   
        "facebook/convnext-base-224"
    },
    "Object-Detection": {
        # "facebook/detr-resnet-101",   Failed to skip
        # "facebook/detr-resnet-50",    Failed to skip
        "hustvl/yolos-tiny",
        "SenseTime/deformable-detr",
        "Aryn/deformable-detr-DocLayNet",
        "nickmuchi/yolos-small-finetuned-license-plate-detection",
        "SenseTime/deformable-detr-with-box-refine",
        "hustvl/yolos-base",
        "microsoft/conditional-detr-resnet-50",
    },
    "Image-Segmentation": {
        "facebook/mask2former-swin-base-coco-panoptic",      #failed 
        "facebook/mask2former-swin-large-ade-semantic",
        "shi-labs/oneformer_coco_swin_large",

    },
    "Image-To-Text": [
        "microsoft/trocr-base-handwritten",
        "nlpconnect/vit-gpt2-image-captioning",
        "Salesforce/blip-image-captioning-base"
    ]
}

image_classification_configutation = {
    "microsoft/resnet-50": 
        [AutoImageProcessor.from_pretrained("microsoft/resnet-50"),
         ResNetForImageClassification.from_pretrained("microsoft/resnet-50")],
    "google/vit-base-patch16-224":
        [ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'),
         ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')],
    "Falconsai/nsfw_image_detection":
        [ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection'),
         AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")],
    "nateraw/vit-age-classifier":
        [ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier'),
         ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')],
    "microsoft/beit-base-patch16-224-pt22k-ft22k":
        [BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k'),
         BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')],
    "microsoft/resnet-18":
        [AutoImageProcessor.from_pretrained("microsoft/resnet-18"),
         AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")],
    "nvidia/mit-b0":
        [SegformerImageProcessor.from_pretrained("nvidia/mit-b0"),
         SegformerForImageClassification.from_pretrained("nvidia/mit-b0")],
    "google/vit-base-patch16-384":
        [ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384'),
         ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')],
    "facebook/convnext-large-224":
        [ConvNextImageProcessor.from_pretrained("facebook/convnext-large-224"),
         ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224")],
    "google/vit-large-patch16-224":
        [ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224'),
         ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')],
    "facebook/deit-base-distilled-patch16-224":
        [AutoFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224'),
         DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-224')],
    "google/vit-base-patch32-384":
        [ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-384'),
         ViTForImageClassification.from_pretrained('google/vit-base-patch32-384')],
    "microsoft/resnet-101":
        [AutoFeatureExtractor.from_pretrained("microsoft/resnet-101"),
         ResNetForImageClassification.from_pretrained("microsoft/resnet-101")],
    "microsoft/cvt-13":
        [AutoFeatureExtractor.from_pretrained('microsoft/cvt-13'),
         CvtForImageClassification.from_pretrained('microsoft/cvt-13')],
    "google/vit-large-patch16-384":
        [ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-384'),
         ViTForImageClassification.from_pretrained('google/vit-large-patch16-384')],
    "microsoft/beit-base-patch16-224":
        [BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224'),
         BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')],
    "microsoft/beit-large-patch16-512":
        [BeitFeatureExtractor.from_pretrained('microsoft/beit-large-patch16-512'),
         BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-512')],
    "microsoft/resnet-152":
        [AutoFeatureExtractor.from_pretrained("microsoft/resnet-152"),
         ResNetForImageClassification.from_pretrained("microsoft/resnet-152")],
    "google/mobilenet_v2_1.0_224":
        [AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224"),
         AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")],
    "google/efficientnet-b7":
        [EfficientNetImageProcessor.from_pretrained("google/efficientnet-b7"),
         EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7")],
    "facebook/convnext-base-224":
        [ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224"),
         ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224")]
}   

object_detection_configutation = {
    "facebook/detr-resnet-101": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm"),
        DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")],
    "facebook/detr-resnet-50": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm"),
        DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")],
    "hustvl/yolos-tiny": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        YolosImageProcessor.from_pretrained("hustvl/yolos-tiny"),
        YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')],
    "SenseTime/deformable-detr": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("SenseTime/deformable-detr"),
        DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")],
    "Aryn/deformable-detr-DocLayNet": [
        "https://huggingface.co/Aryn/deformable-detr-DocLayNet/resolve/main/examples/doclaynet_example_1.png",
        AutoImageProcessor.from_pretrained("Aryn/deformable-detr-DocLayNet"),
        DeformableDetrForObjectDetection.from_pretrained("Aryn/deformable-detr-DocLayNet")],
    "nickmuchi/yolos-small-finetuned-license-plate-detection": [
        'https://drive.google.com/uc?id=1p9wJIqRz3W50e2f_A0D8ftla8hoXz4T5',
        YolosFeatureExtractor.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection'),
        YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-finetuned-license-plate-detection')],
    "SenseTime/deformable-detr-with-box-refine": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("SenseTime/deformable-detr-with-box-refine"),
        DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr-with-box-refine")],
    "hustvl/yolos-base": [
        'http://images.cocodataset.org/val2017/000000039769.jpg',
        YolosFeatureExtractor.from_pretrained('hustvl/yolos-base'),
        YolosForObjectDetection.from_pretrained('hustvl/yolos-base')],
    "microsoft/conditional-detr-resnet-50": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50"),
        ConditionalDetrForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")],
    
}

Image_Segmentation_configuration = {
    "facebook/mask2former-swin-base-coco-panoptic": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic"),
        Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")],
    "facebook/mask2former-swin-large-ade-semantic": [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic"),
        Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")],
    "shi-labs/oneformer_coco_swin_large": [
        "https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/coco.jpeg",
        OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large"),
        OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")
    ]
}

def test_text_generation_eval():
    for model_id in model_dict["Text_Generation"]:
        pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda")
        pipeline("Hey how are you doing today?")


def test_text_classification_eval():
    for model_id in model_dict["Text-classification"]:
        pipeline = transformers.pipeline("sentiment-analysis", model= model_id, device=cuda_device)
        result = pipeline("Hello, world")[0]
        # import pdb;pdb.set_trace()
        print("Text classification model name {}, result: {}".format(model_id, result))


def test_image_classification_eval():
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    for model_id in model_dict["Image-Classification"]:
        processor, model = image_classification_configutation[model_id]
        model = model.to(cuda_device)

        inputs = processor(image, return_tensors="pt", device=cuda_device)

        inputs["pixel_values"] = inputs["pixel_values"].to(cuda_device)
        with torch.no_grad():
            logits = model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        print("-" * 60)
        print("Testing model name:{}, result device:{}".format(model_id, logits.device.type))
        print("Testing image classification model name {}, result:{}".format(model_id, model.config.id2label[predicted_label]))


def test_object_detection_eval():
    
    for model_id in model_dict["Object-Detection"]:
        url, processor, model = object_detection_configutation[model_id]
        image = Image.open(requests.get(url, stream=True).raw)
        model = model.to(cuda_device)
        inputs = processor(images=image, return_tensors="pt", device=cuda_device)
        
        for k in inputs.keys():
            inputs[k] = inputs[k].to(cuda_device)

        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        print("-" * 60)
        print("Testing model name:{}, result device:{}".format(model_id, results["scores"].device.type))
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
            )


def test_image_segmentation_eval():

    for model_id in model_dict["Image-Segmentation"]:
        url, processor, model = Image_Segmentation_configuration[model_id]
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(images=image, return_tensors="pt", device=cuda_device)
        model = model.to(cuda_device)

        for k in inputs.keys():
            inputs[k] = inputs[k].to(cuda_device)

        with torch.no_grad():
            outputs = model(**inputs)

        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_panoptic_map = result["segmentation"]
        print("-" * 60) 
        print("Testing image_segmentation for model: {}, result device:{}, result:{}".format(model_id, predicted_panoptic_map.device.type, predicted_panoptic_map))
        print(predicted_panoptic_map)


image_to_text_configuration = {
    "microsoft/trocr-base-handwritten": [
        'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg',
        TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'),
        VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')],
    "nlpconnect/vit-gpt2-image-captioning": [
        'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg',
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"),
        BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")],
    "Salesforce/blip-image-captioning-base": [
        'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg',
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"),
        BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")]
}

def test_Image_To_Text_eval():
    for model_id in model_dict["Image-To-Text"]:

        url, processor, model = image_to_text_configuration[model_id]
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt", device=cuda_device).pixel_values
        pixel_values = pixel_values.to(cuda_device)
        model = model.to(cuda_device)

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("-" * 60)
        print("Testing model:{}, result:{}".format(model_id, generated_text))


if __name__ == "__main__":
    # test_text_classification_eval()
    # test_image_classification_eval()
    # test_object_detection_eval()
    # test_image_segmentation_eval()
    test_Image_To_Text_eval()