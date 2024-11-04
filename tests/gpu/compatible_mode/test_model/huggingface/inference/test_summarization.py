import torch
import intel_extension_for_pytorch as ipex

ipex.compatible_mode()

from transformers import pipeline

cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model_dict = [
    # "Summarization": {
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6",
    "philschmid/bart-large-cnn-samsum",
    "google/pegasus-xsum",
    "Falconsai/text_summarization",
    "suriya7/bart-finetuned-text-summarization",
    "human-centered-summarization/financial-summarization-pegasus",
    "cointegrated/rut5-base-absum",
    "Einmalumdiewelt/T5-Base_GNAD",
    "google/bigbird-pegasus-large-arxiv",
    "sshleifer/distilbart-cnn-6-6",
    "google/pegasus-cnn_dailymail",
    "jotamunz/billsum_tiny_summarization",
    "facebook/bart-large-xsum",
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

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County
, New York. A year later, she got married again in Westchester County, 
but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. 
Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. 
In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree,"
referring to her false statements on the 2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, 
Christopher Wright, who declined to comment further. After leaving court, 
Barrientos was arrested and charged with theft of service
 and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times,
 with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. 
She is believed to still be married to four men, and at one time, 
she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, 
who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. 
It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement
 and the Department of Homeland Security\'s Investigation Division. 
 Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his 
native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  
Her next court appearance is scheduled for May 18.
"""


def test_summarization_eval():
    for model_id in model_dict:

        summarizer = pipeline("summarization", model=model_id, device=0)
        if args.precision == "fp16":
            summarizer.model = summarizer.model.to(torch.float16)
        elif args.precision == "bf16":
            summarizer.model = summarizer.model.to(torch.bfloat16)
        if args.backend == "torch_compile":
            model = torch.compile(summarizer.model)
            summarizer.model = model
        result = summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)

        print("*" * 60)
        print("Test result:", result)
        print("Testing model:", model_id)
        print("result device:", summarizer.device)
        print(f"Testing model success: {model_id}")
        # except Exception as e:
        #    print(f"Testing model fail: {model_id}")
        #    print(f"Testing model fail reason: {e}")


if __name__ == "__main__":
    test_summarization_eval()
