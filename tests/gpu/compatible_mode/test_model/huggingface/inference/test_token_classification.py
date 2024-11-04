# encoding: UTF-8
import torch
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import BertTokenizerFast
from transformers import pipeline, AutoModel

cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model_dict = [
    # "Token-Classification": {
    "dslim/bert-base-NER",
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
    "blaze999/Medical-NER",
    "ckiplab/bert-base-chinese-ner",
    "Babelscape/wikineural-multilingual-ner",
    "dslim/bert-large-NER",
    "xlm-roberta-large-finetuned-conll03-english",
    "KoichiYasuoka/bert-base-thai-upos",
    "pierreguillou/ner-bert-base-cased-pt-lenerbr",
    "Davlan/bert-base-multilingual-cased-ner-hrl",
    "KB/bert-base-swedish-cased",
    "ckiplab/albert-tiny-chinese-ws",
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

token_classification_configuration = {
    "dslim/bert-base-NER": [
        "My name is Wolfgang and I live in Berlin",
        AutoTokenizer.from_pretrained("dslim/bert-base-NER"),
        AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER"),
    ],
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english": [
        "My name is Wolfgang and I live in Berlin",
        AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english"),
        AutoModelForTokenClassification.from_pretrained(
            "xlm-roberta-large-finetuned-conll03-english"
        ),
    ],
    "blaze999/Medical-NER": [
        "45 year old woman diagnosed with CAD",
        AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER"),
        AutoModelForTokenClassification.from_pretrained(
            "Clinical-AI-Apollo/Medical-NER"
        ),
    ],
    "ckiplab/bert-base-chinese-ner": [
        "英子是个好公司！",
        BertTokenizerFast.from_pretrained("bert-base-chinese"),
        AutoModel.from_pretrained("ckiplab/bert-base-chinese-ner"),
    ],
    "Babelscape/wikineural-multilingual-ner": [
        "My name is Wolfgang and I live in Berlin",
        AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner"),
        AutoModelForTokenClassification.from_pretrained(
            "Babelscape/wikineural-multilingual-ner"
        ),
    ],
    "dslim/bert-large-NER": [
        "My name is Wolfgang and I live in Berlin",
        AutoTokenizer.from_pretrained("dslim/bert-large-NER"),
        AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER"),
    ],
    "xlm-roberta-large-finetuned-conll03-english": [
        "My name is Wolfgang and I live in Berlin",
        AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english"),
        AutoModelForTokenClassification.from_pretrained(
            "xlm-roberta-large-finetuned-conll03-english"
        ),
    ],
    "KoichiYasuoka/bert-base-thai-upos": [
        "My name is Wolfgang and I live in Berlin",
        AutoTokenizer.from_pretrained("KoichiYasuoka/bert-base-thai-upos"),
        AutoModelForTokenClassification.from_pretrained(
            "KoichiYasuoka/bert-base-thai-upos"
        ),
    ],
    "pierreguillou/ner-bert-base-cased-pt-lenerbr": [
        "My name is Wolfgang and I live in Berlin",
        AutoTokenizer.from_pretrained("pierreguillou/ner-bert-base-cased-pt-lenerbr"),
        AutoModelForTokenClassification.from_pretrained(
            "pierreguillou/ner-bert-base-cased-pt-lenerbr"
        ),
    ],
    "Davlan/bert-base-multilingual-cased-ner-hrl": [
        "My name is Wolfgang and I live in Berlin",
        AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl"),
        AutoModelForTokenClassification.from_pretrained(
            "Davlan/bert-base-multilingual-cased-ner-hrl"
        ),
    ],
    "KB/bert-base-swedish-cased": [
        "My name is Wolfgang and I live in Berlin",
        AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased"),
        AutoModel.from_pretrained("KB/bert-base-swedish-cased"),
    ],
    "ckiplab/albert-tiny-chinese-ws": [
        "英子是个好公司！",
        BertTokenizerFast.from_pretrained("bert-base-chinese"),
        AutoModel.from_pretrained("ckiplab/albert-tiny-chinese-ws"),
    ],
}


def test_one(model_id):
    example, tokenizer, model = token_classification_configuration[model_id]

    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)
    if args.precision == "fp16":
        nlp.model = nlp.model.to(torch.float16)
    elif args.precision == "bf16":
        nlp.model = nlp.model.to(torch.bfloat16)
    if args.backend == "torch_compile":
        nmodel = torch.compile(nlp.model)
        nlp.model = nmodel
    result = summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)

    ner_results = nlp(example)
    print("*" * 60)
    print("test result:", ner_results)
    print("testing model:", model_id)
    print("testing device:", nlp.device)


def test_token_classificiation_eval():
    for model_id in model_dict:
        # try:
        test_one(model_id)
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_token_classificiation_eval()
