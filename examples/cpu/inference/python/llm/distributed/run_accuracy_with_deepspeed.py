import os
import argparse
import json
import re
import torch
from pathlib import Path
import intel_extension_for_pytorch as ipex
from tqdm import tqdm
import math
import torch.nn.functional as F
import re
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from huggingface_hub import snapshot_download
from transformers.utils import is_offline_mode
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    T5ForConditionalGeneration,
)

import sys

sys.path.append(sys.path[0] + '/../../')

MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, LlamaTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "codegen": (AutoModelForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "chatglm": (AutoModelForCausalLM, AutoTokenizer),
    "gptbigcode": (AutoModelForCausalLM, AutoTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "mistral": (AutoModelForCausalLM, AutoTokenizer),
    "mixtral": (AutoModelForCausalLM, AutoTokenizer),
    "mpt": (AutoModelForCausalLM, AutoTokenizer),
    "stablelm": (AutoModelForCausalLM, AutoTokenizer),
    "qwen": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="?", default="EleutherAI/gpt-j-6b")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--device", default="cpu", type=str, help="cpu")
parser.add_argument(
    "--dtype", default="bfloat16", type=str, help="float32 or bfloat16 or int8"
)
parser.add_argument(
    "--batch-size", default=1, type=int, help="For accuracy measurement only."
)
parser.add_argument(
    "--save-accuracy-path", default=None, help="Save accuracy results path."
)
parser.add_argument(
    "--ipex", action="store_true", help="use intel extension for pytorch."
)
parser.add_argument(
    "--disable-jit", action="store_true", help="disable converting model to torchscript mode."
)
parser.add_argument("--quant-with-amp", action="store_true", help="by default static quant is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)")
parser.add_argument("--quantized-model-path", default="./saved_result/best_model.pt")
parser.add_argument(
    "--tasks",
    nargs="+",
    default=[
        "lambada_standard",
    ],
    type=str,
    help="tasks list for accuracy validation, only enabled lambada_standard and lambada_standard at present",
)
parser.add_argument(
    "--local_rank", required=False, type=int, help="used by dist launchers"
)
parser.add_argument(
    "--config-file", default=None, type=str, help="specific configuration file"
)
parser.add_argument(
    "--ipex-weight-only-quantization",
    action="store_true",
    help="use ipex weight-only quantization",
)
parser.add_argument(
    "--lowp-mode",
    choices=["AUTO", "BF16", "FP32", "INT8", "FP16"],
    default="AUTO",
    type=str,
    help="low precision mode for weight only quantization. "
         "It indicates data type for computation for speedup at the cost "
         "of accuracy. Unrelated to activation or weight data type."
         "It is not supported yet to use lowp_mode=INT8 for INT8 weight, "
         "falling back to lowp_mode=BF16 implicitly in this case."
         "If set to AUTO, lowp_mode is determined by weight data type: "
         "lowp_mode=BF16 is used for INT8 weight "
         "and lowp_mode=INT8 used for INT4 weight",
)
parser.add_argument(
    "--weight-dtype",
    choices=["INT8", "INT4"],
    default="INT8",
    type=str,
    help="weight data type for weight only quantization. Unrelated to activation data type or lowp-mode.",
)
parser.add_argument(
    "--group-size",
    default=-1,
    type=int,
    help="For weight-only quantization only. Specifies the group size along"
    " input channel for block-wise quantization of weight. It must be a"
    " positive power of 2 or -1. If it is -1, weight is quantized per"
    " output channel. Otherwise, weight is quantized per block with block size"
    " = [1, group_size]. If `--low-precision-checkpoint` is given, group"
    " size is determined automatically and this argument has no effect.",
)
parser.add_argument(
    "--act-quant-mode",
    choices=["PER_TENSOR", "PER_IC_BLOCK", "PER_BATCH", "PER_BATCH_IC_BLOCK"],
    default="PER_IC_BLOCK",
    type=str,
    help="Quantization mode for activation with different granularity. "
    "For lowp-mode=INT8 only. For other cases, it has no effect. "
    "Assume the activation tensor has shape batch_size x input_channel. "
    "PER_TENSOR(0): quantize per tensor; "
    "PER_IC_BLOCK(1): quantize per group along IC with group size = IC_BLOCK; "
    "PER_BATCH(2): quantize per batch; "
    "PER_BATCH_IC_BLOCK(3): quantize per block of size 1 x IC_BLOCK. "
    "IC_BLOCK is determined by IC automatically.",
)

args = parser.parse_args()

use_ipex = args.ipex or args.ipex_weight_only_quantization

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


local_rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID"], "0")
world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE"], "1")

deepspeed.init_distributed(get_accelerator().communication_backend_name())

print("init_distributed done")


import lm_eval
from lm_eval import evaluator, utils
from lm_eval.base import BaseLM
from typing import Union, List, Optional, Tuple
from transformers import BatchEncoding
import transformers

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

class HuggingFaceModel(BaseLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        device="cpu",
        with_ipex=True,
        with_jit=True,
        with_greedy=False,
        batch_size=1,
        max_length=None,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        tp_number=1,
        config=None,
        add_special_tokens = True,
    ):
        super().__init__()

        model_id = pretrained
        self._device = device
        self._batch_size = batch_size
        self._with_jit = with_jit
        self._with_ipex = with_ipex
        self._with_greedy = with_greedy
        self._max_length = max_length
        self._dtype = dtype
        self._tp_number = tp_number
        self.add_special_tokens = add_special_tokens

        load_dtype = torch.float32
        infer_dtype = torch.float32
        if args.quant_with_amp or dtype == "bfloat16":
            load_dtype = torch.bfloat16
            infer_dtype = torch.bfloat16
        else:
            if dtype == "float16":
                load_dtype = torch.half
                infer_dtype = torch.half
            elif dtype == "int8":
                load_dtype = torch.float32
                infer_dtype = torch.int8

        model_type = next(
            (x for x in MODEL_CLASSES.keys() if x in model_id.lower()), "auto"
        )
        model_class = MODEL_CLASSES[model_type]

        self.tokenizer = model_class[1].from_pretrained(
            model_id, trust_remote_code=True
        )
        self.config = AutoConfig.from_pretrained(
            model_id if config is None else config, torchscript=with_jit, trust_remote_code=True
        )

        if re.search("gptbigcode", self.config.architectures[0], re.IGNORECASE):
            model_type = "gptbigcode"

        # For now, Falcon, baichuan and gptbigcode have accuracy issue with from_config with deepspeed meta device load.
        # TODO: we will change the scope once deepspeed providing the support
        if world_size == 1 or model_type in ["falcon", "baichuan", "gptbigcode", "qwen"]:
            self.model = model_class[0].from_pretrained(
                model_id,
                config=self.config,
                low_cpu_mem_usage=True,
                torch_dtype=load_dtype,
                trust_remote_code=True,
            )
        else:
            with deepspeed.OnDevice(dtype=load_dtype, device="meta"):
                if model_type in ["t5"]:
                    self.model =  model_class[0](config=self.config)
                else:
                    if model_class[0] == AutoModelForCausalLM:
                        self.model = (
                            model_class[0]
                            .from_config(self.config, trust_remote_code=True)
                            .to(load_dtype)
                        )
                    else:
                        self.model = model_class[0].from_pretrained(
                            model_id,
                            low_cpu_mem_usage=True,
                            config=self.config,
                            torch_dtype=load_dtype,
                            trust_remote_code=True,
                        )

        self.model = self.model.eval()

        checkpoints_json = "checkpoints.json"

        def print_rank0(*msg):
            if local_rank != 0:
                return
            print(*msg)

        def get_repo_root(model_name_or_path):
            if os.path.exists(model_name_or_path):
                # local path
                return model_name_or_path
            # checks if online or not
            if is_offline_mode():
                print_rank0("Offline mode: forcing local_files_only=True")
            # download only on first process
            allow_patterns = ["*.bin", "*.model", "*.json", "*.txt", "*.py", "*LICENSE"]
            if local_rank == 0:
                snapshot_download(
                    model_name_or_path,
                    local_files_only=is_offline_mode(),
                    cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                    allow_patterns=allow_patterns,
                    # ignore_patterns=["*.safetensors"],
                )

            dist.barrier()

            return snapshot_download(
                model_name_or_path,
                local_files_only=is_offline_mode(),
                cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                allow_patterns=allow_patterns,
                # ignore_patterns=["*.safetensors"],
            )

        def get_checkpoint_files(model_name_or_path):
            cached_repo_dir = get_repo_root(model_name_or_path)

            # extensions: .bin | .pt
            # creates a list of paths from all downloaded files in cache dir
            file_list = [
                str(entry)
                for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]")
                if entry.is_file()
            ]
            return file_list

        def write_checkpoints_json():
            checkpoint_files = get_checkpoint_files(model_id)
            if local_rank == 0:
                # model.config.model_type.upper()
                data = {
                    "type": "BLOOM",
                    "checkpoints": checkpoint_files,
                    "version": 1.0,
                }
                json.dump(data, open(checkpoints_json, "w"))

        repo_root = get_repo_root(model_id)
        write_checkpoints_json()
        dist.barrier()
        self.model = deepspeed.init_inference(
            self.model,
            mp_size=tp_number,
            base_dir=repo_root,
            dtype=infer_dtype,
            checkpoint=checkpoints_json,
        )

        self.model = self.model.module

        if self._with_ipex:
            ipex_woq_enabled = args.ipex_weight_only_quantization
            if ipex_woq_enabled:
                from intel_extension_for_pytorch.quantization import WoqWeightDtype
                weight_dtype = (
                    WoqWeightDtype.INT4 if args.weight_dtype == "INT4" else WoqWeightDtype.INT8
                )

                if args.lowp_mode == "INT8":
                    lowp_mode = ipex.quantization.WoqLowpMode.INT8
                elif args.lowp_mode == "FP32":
                    lowp_mode = ipex.quantization.WoqLowpMode.NONE
                elif args.lowp_mode == "FP16":
                    lowp_mode = ipex.quantization.WoqLowpMode.FP16
                elif args.lowp_mode == "BF16":
                    lowp_mode = ipex.quantization.WoqLowpMode.BF16
                else:  # AUTO
                    if weight_dtype == WoqWeightDtype.INT4:
                        lowp_mode = ipex.quantization.WoqLowpMode.INT8
                    else:
                        lowp_mode = ipex.quantization.WoqLowpMode.BF16

                act_quant_mode_dict = {
                    "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
                    "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                    "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
                    "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
                }
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    weight_dtype=weight_dtype,
                    lowp_mode=lowp_mode,
                    act_quant_mode=act_quant_mode_dict[args.act_quant_mode],
                    group_size=args.group_size,
                )
            self.model = ipex.llm.optimize(
                self.model.eval(),
                dtype=infer_dtype,
                quantization_config=qconfig if ipex_woq_enabled else None,
                inplace=True,
                deployment_mode=False,
            )

        self.base_model = self.model

        self.num_beams = 1 if with_greedy else 4
        self.iter = 0
        self.is_t5 = re.search("t5", self.base_model.config.architectures[0], re.IGNORECASE)

    def _get_target_nums(self, names):
        for n in names:
            if hasattr(self.base_model.config, n):
                return getattr(self.base_model.config, n)
        print(f"Not found target {names[0]}")
        exit(0)

    def _get_past_key_values(self, input_bs, last_hidden_state=None):
        num_heads_names = ["num_attention_heads", "n_head", "num_heads", "n_heads"]
        num_layers_names = ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]
        hidden_size_names = ["hidden_size", "n_embd"]
        num_attention_heads = self._get_target_nums(num_heads_names)
        num_hidden_layers = self._get_target_nums(num_layers_names)
        hidden_size = self._get_target_nums(hidden_size_names)

        beam_idx_tmp = torch.zeros((2048, int(input_bs)), dtype=torch.long).contiguous()
        num_heads = int(num_attention_heads / self._tp_number)
        head_dim = int(hidden_size / num_attention_heads)
        past_key_values = tuple(
            [
                (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, num_heads, 1, head_dim]).contiguous(),
                    torch.zeros([1, num_heads, 1, head_dim]).contiguous(),
                    beam_idx_tmp,
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    self.base_model.decoder.block[i].layer[1].EncDecAttention.k(last_hidden_state).view(
                        int(input_bs),
                        -1,
                        self.base_model.decoder.block[i]
                        .layer[1]
                        .EncDecAttention.n_heads,
                        self.base_model.decoder.block[i]
                        .layer[1]
                        .EncDecAttention.key_value_proj_dim,
                    ).transpose(0, 1).contiguous(),
                    self.base_model.decoder.block[i].layer[1].EncDecAttention.v(last_hidden_state).view(
                        int(input_bs),
                        -1,
                        self.base_model.decoder.block[i]
                        .layer[1]
                        .EncDecAttention.n_heads,
                        self.base_model.decoder.block[i]
                        .layer[1]
                        .EncDecAttention.key_value_proj_dim,
                    ).transpose(0, 1).contiguous(),
                    beam_idx_tmp,
                ) if self.is_t5 else
                (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, num_heads, 1, head_dim]).contiguous(),
                    torch.zeros([1, num_heads, 1, head_dim]).contiguous(),
                    beam_idx_tmp,
                )
                for i in range(num_hidden_layers)
            ]
        )
        return past_key_values

    def _model_call(
        self,
        inputs: TokenSequence,
        labels: Optional[TokenSequence] = None,
        past_key_values: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> TokenSequence:
        _attention_mask = []
        _position_ids = []
        if self.is_t5:
            inputs = inputs['input_ids']
        for text in inputs:
            input_ids = text.to(self._device)
            input_bs = inputs.shape[0] * self.num_beams
            position_ids = torch.arange(len(input_ids))
            attention_mask = torch.ones(len(input_ids))
            _attention_mask.append(attention_mask)
            _position_ids.append(position_ids)

        attention_mask_batched = torch.stack(_attention_mask)
        position_ids_batched = torch.stack(_position_ids)
        if self.is_t5:
            model_kwargs = {"attention_mask": attention_mask_batched}
            model_kwargs = self.base_model._prepare_encoder_decoder_kwargs_for_generation(inputs, model_kwargs, "input_ids")
            (
                inputs,
                example_inputs,
            ) = self.base_model._expand_inputs_for_generation(
                input_ids=inputs,
                expand_size=self.num_beams,
                is_encoder_decoder=True,
                **model_kwargs,
            )
            past_key_values = self._get_past_key_values(input_bs, example_inputs["encoder_outputs"]["last_hidden_state"])
            if self.num_beams == 1:
                decoder_input_ids = self.base_model._shift_right(labels['input_ids'])
                _labels = labels['input_ids']
            else:
                decoder_input_ids = self.base_model._shift_right(labels['input_ids'].repeat_interleave(self.num_beams, dim=0))
                _labels = labels['input_ids'].repeat_interleave(self.num_beams, dim=0)
            example_dict = {
                    "decoder_input_ids": decoder_input_ids,
                    "encoder_outputs": (example_inputs["encoder_outputs"]["last_hidden_state"],),
                    "labels": _labels,
                }
        else:
            past_key_values = self._get_past_key_values(input_bs)
            example_dict = {"input_ids": inputs}
        model_inputs = self.base_model.prepare_inputs_for_generation(inputs, attention_mask=attention_mask_batched)
        has_position_ids = model_inputs.get("position_ids", None) is not None
        if self._with_jit:
            example_dict["attention_mask"]= attention_mask_batched
            example_dict["past_key_values"]= past_key_values
            example_dict["return_dict"] = torch.tensor(False)
            if has_position_ids:
                example_dict["position_ids"] = position_ids_batched
        if "return_last_logit" in model_inputs:
            example_dict["return_last_logit"] = torch.tensor(True)

        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=True
            if args.quant_with_amp or self._dtype == "bfloat16"
            else False,
        ):
            if self._with_jit and self.iter == 0:
                if self._dtype != "int8":
                    self.model = torch.jit.trace(
                        self.model.eval(),
                        example_kwarg_inputs=example_dict,
                        strict=False,
                        check_trace=False,
                    )
                    self.model = torch.jit.freeze(self.model.eval())
                else:
                    self.model = torch.jit.load(args.quantized_model_path)
                    self.model = torch.jit.freeze(self.model.eval())

                self.model(**example_dict)
                self.model(**example_dict)
                self.iter = self.iter + 1

            output = self.model(**example_dict)

        if isinstance(output, tuple):
            if labels is not None:
                return output[1]
            return output[0]

        return output["logits"]

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length

        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self._batch_size  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            # setting eos_token_id as pad token
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs["pad_token_id"] = eos_token_id
        return self.model.generate(context, **generation_kwargs)

class HuggingFaceSeq2SeqModel(HuggingFaceModel):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        new_requests = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)

            # Fill empty contexts with the EOT token.
            context = [
                f"{self.eot_token}" if len(text) == 0 else text for text in context
            ]
            context_enc = self.tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -self.max_length :]

            # Remove leading whitespace introduced by the default
            # `text_target_separator` since the context and continuation
            # will not be concatenated as a single (decoder) input.
            continuation = [text.lstrip() for text in continuation]
            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -self.max_length :]

            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
        return self._loglikelihood_tokens(new_requests)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            contexts, conts = utils.split_and_pad_windows(
                rolling_token_windows,
                pad_token_id=self.eot_token_id,
                max_seq_len=self.max_length,
            )
            # Manually create BatchEncoding tensors with attention masks as
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            contexts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": contexts_enc,
                    "attention_mask": (contexts_enc != self.eot_token_id).long(),
                }
            )
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": conts_enc,
                    "attention_mask": (conts_enc != self.eot_token_id).long(),
                }
            )
            # TODO: Extract out this call so it only gets called once and also
            # somehow figure out partial caching for.
            rolling_token_windows_request = [
                ((contexts, conts), contexts_enc, conts_enc)
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows_request, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:

        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results

class T5ModelLambada(HuggingFaceSeq2SeqModel):
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:

        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)

            outputs = self._model_call(inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )

            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                if length >= 1 and target_tokens[length-1].item() == self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens = False)[0]:
                    length = length - 1

                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_text = self.tokenizer.decode(target_tokens, skip_special_tokens = True)
                greedy_text = self.tokenizer.decode(greedy_tokens, skip_special_tokens = True)
                max_text_equal = (greedy_text == target_text)
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal or max_text_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results


task_dict = lm_eval.tasks.get_task_dict(args.tasks)
torch._C._jit_set_texpr_fuser_enabled(False)
if args.model in ["google/flan-t5-xl"]:
    hfmodel = T5ModelLambada(
        pretrained=args.model,
        device="cpu",
        batch_size=args.batch_size,
        with_ipex=use_ipex,
        with_jit=not args.disable_jit,
        dtype=args.dtype,
        tp_number=world_size,
        config=args.config_file,
        add_special_tokens=True,
        with_greedy=False,
    )
else:
    hfmodel = HuggingFaceModel(
        pretrained=args.model,
        device="cpu",
        batch_size=args.batch_size,
        with_ipex=use_ipex,
        with_jit=not args.disable_jit,
        dtype=args.dtype,
        tp_number=world_size,
        config=args.config_file,
        add_special_tokens=False
    )

results = evaluator.evaluate(
    hfmodel,
    task_dict,
    #        bootstrap_iters=1000,
    #        limit=100
)

print(evaluator.make_table(results))
