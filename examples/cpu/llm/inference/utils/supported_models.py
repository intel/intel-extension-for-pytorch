from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    WhisperForConditionalGeneration,
    MllamaForConditionalGeneration,
    AutoProcessor,
)

# supported models
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, AutoTokenizer),
    "mllama": (MllamaForConditionalGeneration, AutoProcessor),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "codegen": (AutoModelForCausalLM, AutoTokenizer),
    "baichuan2": (AutoModelForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "chatglm": (AutoModelForCausalLM, AutoTokenizer),
    "gptbigcode": (AutoModelForCausalLM, AutoTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "mixtral": (AutoModelForCausalLM, AutoTokenizer),
    "mistral": (AutoModelForCausalLM, AutoTokenizer),
    "mpt": (AutoModelForCausalLM, AutoTokenizer),
    "stablelm": (AutoModelForCausalLM, AutoTokenizer),
    "qwen": (AutoModelForCausalLM, AutoTokenizer),
    "git": (AutoModelForCausalLM, AutoProcessor),
    "yuan": (AutoModelForCausalLM, AutoTokenizer),
    "phi-3": (AutoModelForCausalLM, AutoTokenizer),
    "phio": (AutoModelForCausalLM, AutoProcessor),
    "phi": (AutoModelForCausalLM, AutoTokenizer),
    "whisper": (WhisperForConditionalGeneration, AutoProcessor),
    "maira2": (AutoModelForCausalLM, AutoProcessor),
    "maira-2": (AutoModelForCausalLM, AutoProcessor),
    "jamba": (AutoModelForCausalLM, AutoTokenizer),
    "deepseek-v2": (AutoModelForCausalLM, AutoTokenizer),
    "deepseek-v3": (AutoModelForCausalLM, AutoTokenizer),
    "deepseekv2": (AutoModelForCausalLM, AutoTokenizer),
    "deepseekv3": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

try:
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    MODEL_CLASSES["llava"] = (LlavaLlamaForCausalLM, AutoTokenizer)
except ImportError:
    pass
