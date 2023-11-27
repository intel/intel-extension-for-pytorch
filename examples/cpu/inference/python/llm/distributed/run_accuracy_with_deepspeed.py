import os
import argparse
import json
import re
import torch
from pathlib import Path
import intel_extension_for_pytorch as ipex

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
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="?", default="EleutherAI/gpt-j-6b")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--device", default="cpu", type=str, help="cpu")
parser.add_argument(
    "--dtype", default="bfloat16", type=str, help="float32 or bfloat16 or int8"
)
parser.add_argument("--accuracy-only", action="store_true")
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
    "--jit", action="store_true", help="convert model to torchscript mode."
)
parser.add_argument("--int8-bf16-mixed", action="store_true", help="int8 mixed bf16")
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

args = parser.parse_args()


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

if args.accuracy_only:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.base import BaseLM
    from typing import Union, List, Optional
    from transformers import BatchEncoding

    TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

    class HuggingFaceModel(BaseLM):
        _DEFAULT_MAX_LENGTH = 2048

        def __init__(
            self,
            device="cpu",
            model_id="",
            with_ipex=True,
            with_jit=True,
            with_greedy=False,
            batch_size=1,
            max_length=None,
            dtype: Optional[Union[str, torch.dtype]] = "auto",
            tp_number=1,
            config=None,
        ):
            super().__init__()

            self._device = device
            self._batch_size = batch_size
            self._with_jit = with_jit
            self._with_ipex = with_ipex
            self._with_greedy = with_greedy
            self._max_length = max_length
            self._dtype = dtype
            self._tp_number = tp_number

            if args.int8_bf16_mixed:
                load_dtype = torch.bfloat16
                infer_dtype = torch.bfloat16
            else:
                if dtype == "float16":
                    load_dtype = torch.half
                    infer_dtype = torch.half
                elif dtype == "bfloat16":
                    load_dtype = torch.bfloat16
                    infer_dtype = torch.bfloat16
                elif dtype == "int8":
                    load_dtype = torch.float32
                    infer_dtype = torch.int8
                elif dtype == "float32":
                    load_dtype = torch.float32
                    infer_dtype = torch.float32

            amp_enabled = True if dtype != "float32" else False
            amp_dtype = getattr(torch, dtype)

            model_type = next(
                (x for x in MODEL_CLASSES.keys() if x in model_id.lower()), "auto"
            )
            model_class = MODEL_CLASSES[model_type]

            self.tokenizer = model_class[1].from_pretrained(
                model_id, trust_remote_code=True
            )
            if config is None:
                self.config = AutoConfig.from_pretrained(
                    model_id, torchscript=with_jit, trust_remote_code=True
                )
            else:
                self.config = AutoConfig.from_pretrained(
                    config, torchscript=with_jit, trust_remote_code=True
                )

            if model_type == "baichuan":
                from llm.utils.utils import _get_relative_imports
                import transformers
                transformers.dynamic_module_utils.get_relative_imports = _get_relative_imports
            if world_size == 1 or model_type == "falcon":
                self.model = model_class[0].from_pretrained(
                    model_id,
                    config=config,
                    low_cpu_mem_usage=True,
                    torch_dtype=load_dtype,
                    trust_remote_code=True,
                )
            else:
                with deepspeed.OnDevice(dtype=load_dtype, device="meta"):
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
                local_prefix = ("/", "./", "../")
                if model_name_or_path.startswith(local_prefix):
                    return model_name_or_path
                # checks if online or not
                if is_offline_mode():
                    print_rank0("Offline mode: forcing local_files_only=True")
                # download only on first process
                allow_patterns = [
                    "*.bin",
                    "*.model",
                    "*.json",
                    "*.txt",
                    "*.py",
                    "*LICENSE",
                ]
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

            self.model = deepspeed.init_inference(
                self.model,
                mp_size=tp_number,
                base_dir=repo_root,
                dtype=infer_dtype,
                checkpoint=checkpoints_json,
            )

            self.model = self.model.module

            if args.ipex:
                ipex_woq_enabled = args.ipex_weight_only_quantization
                if ipex_woq_enabled:
                    weight_dtype = (
                        torch.quint4x2 if args.weight_dtype == "INT4" else torch.qint8
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
                        if weight_dtype == torch.quint4x2:
                            lowp_mode = ipex.quantization.WoqLowpMode.INT8
                        else:
                            lowp_mode = ipex.quantization.WoqLowpMode.BF16

                    qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                        weight_dtype=weight_dtype, lowp_mode=lowp_mode
                    )
                self.model = ipex.optimize_transformers(
                    self.model.eval(),
                    dtype=infer_dtype,
                    quantization_config=qconfig if ipex_woq_enabled else None,
                    inplace=True,
                    deployment_mode=False,
                )

            self.base_model = self.model

            self.num_beams = 1 if with_greedy else 4
            self.iter = 0

        def _model_call(
            self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
        ) -> TokenSequence:
            _attention_mask = []
            _position_ids = []

            model_inputs = self.base_model.prepare_inputs_for_generation(torch.ones(32).to(torch.long))
            has_position_ids = "position_ids" in model_inputs
            if self._with_jit:
                for text in inputs:
                    input_ids = text.to(self._device)
                    input_bs = inputs.shape[0] * self.num_beams
                    beam_idx_tmp = torch.zeros(
                        (2048, int(input_bs)), dtype=torch.long
                    ).contiguous()
                    if hasattr(self.base_model.config, "n_head"):
                        num_attention_heads = self.base_model.config.n_head
                    elif hasattr(self.base_model.config, "num_attention_heads"):
                        num_attention_heads = self.base_model.config.num_attention_heads
                    
                    if hasattr(self.base_model.config, "num_hidden_layers"):
                        num_hidden_layers = self.base_model.config.num_hidden_layers
                    elif hasattr(self.base_model.config, "n_layer"):
                        num_hidden_layers = self.base_model.config.n_layer

                    if hasattr(self.base_model.config, "n_embd"):
                        hidden_size = self.base_model.config.n_embd
                    elif hasattr(self.base_model.config, "hidden_size"):
                        hidden_size = self.base_model.config.hidden_size
                    past_key_values = tuple(
                        [
                            (
                                torch.zeros(
                                    1, 0, 0, 1, dtype=torch.long
                                ).contiguous(),
                                torch.zeros(
                                    [
                                        1,
                                        int(num_attention_heads / self._tp_number),
                                        1,
                                        int(hidden_size / num_attention_heads),
                                    ]
                                ).contiguous(),
                                torch.zeros(
                                    [
                                        1,
                                        int(num_attention_heads / self._tp_number),
                                        1,
                                        int(hidden_size / num_attention_heads),
                                    ]
                                ).contiguous(),
                                beam_idx_tmp,
                            )
                            for i in range(num_hidden_layers)
                        ]
                    )

                    position_ids = torch.arange(len(input_ids))
                    attention_mask = torch.ones(len(input_ids))

                    _attention_mask.append(attention_mask)
                    _position_ids.append(position_ids)

                attention_mask_batched = torch.stack(_attention_mask)
                position_ids_batched = torch.stack(_position_ids)

            if self._with_jit and self.iter == 0:
                with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=True
                    if args.int8_bf16_mixed or self._dtype == torch.bfloat16
                    else False,
                    dtype=torch.bfloat16,
                ):
                    if self._dtype != "int8":
                        if not has_position_ids:
                            example_dict = {
                                "input_ids": inputs,
                                "attention_mask": attention_mask_batched,
                                "past_key_values": past_key_values,
                            }
                        else:
                            example_dict = {
                                "input_ids": inputs,
                                "attention_mask": attention_mask_batched,
                                "position_ids": position_ids_batched,
                                "past_key_values": past_key_values,
                            }

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

                    if not has_position_ids:
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                        )
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                        )
                    else:
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                            position_ids=position_ids_batched,
                        )
                        self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                            position_ids=position_ids_batched,
                        )

                self.iter = self.iter + 1

            with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True
                if args.int8_bf16_mixed or self._dtype == torch.bfloat16
                else False,
                dtype=torch.bfloat16,
            ):
                if self._with_jit:
                    if not has_position_ids:
                        output = self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                        )
                    else:
                        output = self.model(
                            inputs,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask_batched,
                            position_ids=position_ids_batched,
                        )
                else:
                    output = self.base_model(
                        inputs,
                    )

            if isinstance(output, tuple):
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
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens)

        def _model_generate(self, context, max_length, eos_token_id):
            generation_kwargs = {"do_sample": False, "max_length": max_length}
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
                generation_kwargs[
                    "pad_token_id"
                ] = eos_token_id  # setting eos_token_id as pad token
            return self.model.generate(context, **generation_kwargs)

    task_dict = lm_eval.tasks.get_task_dict(args.tasks)
    torch._C._jit_set_texpr_fuser_enabled(False)
    hfmodel = HuggingFaceModel(
        model_id=args.model,
        device="cpu",
        batch_size=args.batch_size,
        with_ipex=args.ipex,
        with_jit=args.jit,
        dtype=args.dtype,
        tp_number=world_size,
        config=args.config_file,
    )

    results = evaluator.evaluate(
        hfmodel,
        task_dict,
        #        bootstrap_iters=1000,
        #        limit=100
    )

    print(evaluator.make_table(results))
