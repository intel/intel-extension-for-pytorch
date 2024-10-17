import os
from threading import Thread
from typing import Iterator

import gradio as gr
import spaces
import torch
import intel_extension_for_pytorch as ipex
import argparse

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

parser = argparse.ArgumentParser("Generation script (fp16/int4 path)", add_help=False)
parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="the huggingface mdoel id",
)
parser.add_argument("--woq", action="store_true", help="use a quantized model",)
parser.add_argument("--woq_checkpoint_path", default="", type=str)
args = parser.parse_args()


DESCRIPTION = """\
# intel XPU chatbot based on Huggingface models
This demo refers to the Huggingface project [Llama-2 7B Chat](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat)
"""

LICENSE = """
<p/>
---
As a derivate work of [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/USE_POLICY.md).
"""

if not torch.xpu.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
else:
    DESCRIPTION += "\n<p>Running on XPU 🚀</p>"


if torch.xpu.is_available() and not args.woq:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    model_id = args.model_id
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").eval().to("xpu").to(memory_format=torch.channels_last)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False
    model = ipex.optimize_transformers(model.eval(), dtype=torch.float16, device="xpu")

elif torch.xpu.is_available() and args.woq:
    from neural_compressor.transformers import AutoModelForCausalLM, RtnConfig
    from transformers import AutoTokenizer, TextIteratorStreamer, AutoConfig
    if args.woq_checkpoint_path:
    # directly load already quantized model
        model = AutoModelForCausalLM.from_pretrained(
            args.woq_checkpoint_path, trust_remote_code=True, device_map="xpu", torch_dtype=torch.float16)
        model = model.to(memory_format=torch.channels_last)
        woq_quantization_config = getattr(model, "quantization_config", None)
        tokenizer = AutoTokenizer.from_pretrained(args.woq_checkpoint_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.woq_checkpoint_path, use_cache=True, # to use kv cache.
                                        trust_remote_code=True)
    else:
        print("Using RTN algorithm quantizing model from huggingface...")
        model_id = args.model_id
        woq_quantization_config = RtnConfig(compute_dtype="fp16", weight_dtype="int4_fullrange", scale_dtype="fp16", group_size=128)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="xpu",
            quantization_config=woq_quantization_config,
            trust_remote_code=True,
            use_llm_runtime=False)
        model = model.to("xpu").to(memory_format=torch.channels_last)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(model)
    tokenizer.use_default_system_prompt = False

    model = ipex.optimize_transformers(model.eval(), device="xpu", inplace=True, quantization_config=woq_quantization_config)

else:
    raise RuntimeError("This demo requires an XPU device to run.")


@spaces.GPU
def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.6,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.2,
        ),
    ],
    stop_btn=None,
    examples=[
        ["Hello there! How are you doing?"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["How many hours does it take a man to eat a Helicopter?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ],
    cache_examples=False,
)

with gr.Blocks(css="style.css", fill_height=True) as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)