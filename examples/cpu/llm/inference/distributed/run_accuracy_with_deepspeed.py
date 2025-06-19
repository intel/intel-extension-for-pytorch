# encoding: UTF-8
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
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from huggingface_hub import snapshot_download
from transformers.utils import is_offline_mode
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import sys
from intel_extension_for_pytorch.llm.utils import (
    load_low_precision_checkpoint,
    shard_low_precision_checkpoint,
)

sys.path.append(sys.path[0] + "/../../../")
from llm.inference.utils.supported_models import MODEL_CLASSES

try:
    import lmms_eval
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import register_model
    from lmms_eval import evaluator as lmms_evaluator
    from lmms_eval import utils as lmms_utils
    from lmms_eval.api.registry import ALL_TASKS
    from lmms_eval.tasks import initialize_tasks
    from llava.model.language_model.llava_llama import (  # noqa F401
        LlavaLlamaForCausalLM,
    )
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.constants import (  # noqa F401
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
except ImportError:

    def register_model(name):
        def decorator(func):
            return func

        return decorator

    from abc import ABC as lmms

    Instance = None
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="?", default="EleutherAI/gpt-j-6b")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--output_path", nargs="?", default="./logs")
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
    "--disable-jit",
    action="store_true",
    help="disable converting model to torchscript mode.",
)
parser.add_argument(
    "--quant-with-amp",
    action="store_true",
    help="by default static quant is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
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
    choices=["INT8", "INT4", "NF4", "FP8"],
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
    choices=[
        "PER_TENSOR",
        "PER_IC_BLOCK",
        "PER_BATCH",
        "PER_BATCH_IC_BLOCK",
        "PER_TENSOR_SYM",
        "PER_IC_BLOCK_SYM",
        "PER_BATCH_SYM",
        "PER_BATCH_IC_BLOCK_SYM",
    ],
    default="PER_BATCH_IC_BLOCK_SYM",
    type=str,
    help="Quantization mode for activation with different granularity. "
    "For lowp-mode=INT8 only. For other cases, it has no effect. "
    "Assume the activation tensor has shape batch_size x input_channel. "
    "PER_TENSOR(0): quantize per tensor; "
    "PER_IC_BLOCK(1): quantize per group along IC with group size = IC_BLOCK; "
    "PER_BATCH(2): quantize per batch; "
    "PER_BATCH_IC_BLOCK(3): quantize per block of size 1 x IC_BLOCK. "
    "PER_TENSOR_SYM(4): symmetrically quantize per tensor; "
    "PER_IC_BLOCK_SYM(5): symmetrically quantize per group along IC with group size = IC_BLOCK; "
    "PER_BATCH_SYM(6): symmetrically quantize per batch; "
    "PER_BATCH_IC_BLOCK_SYM(7): symmetrically quantize per block of size 1 x IC_BLOCK. "
    "IC_BLOCK is determined by IC automatically.",
)
parser.add_argument(
    "--cache-weight-for-large-batch",
    action="store_true",
    help="Cache an extra linear weight for large batch inference, such as the first token (prefill phase)."
    " It brings better performance at the cost of higher memory usage. It is only valid for full bf16 path"
    " and weight-only quantization with lowp-mode=BF16. Otherwise, it has no effect.",
)
parser.add_argument(
    "--woq-sym-quant-weight",
    action="store_true",
    help="Quantize weight symmetrically for weight only quantization. It usually brings better latency at"
    " the cost of accuracy. It has not effect if you are loading low-precision checkpoints.",
)
parser.add_argument(
    "--low-precision-checkpoint",
    default="",
    type=str,
    help="Low precision checkpoint file generated by algorithms, such as GPTQ. It contains"
    " INT4 weights, scales, zero points, etc. For better accuracy of weight only"
    " quantization with INT4 weight.",
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
ds_init_inf_kwargs = {}


def get_low_precision_checkpoint(args, model_config):
    low_precision_checkpoint = None
    quant_config = None
    # Users gives the model by path to int4 checkpoint directory
    if (
        not args.low_precision_checkpoint
        and hasattr(model_config, "quantization_config")
        and os.path.isdir(args.model)
    ):
        args.low_precision_checkpoint = args.model
    if args.ipex_weight_only_quantization and args.low_precision_checkpoint != "":
        pathname = args.low_precision_checkpoint
        low_precision_checkpoint, quant_config = load_low_precision_checkpoint(pathname)
    return low_precision_checkpoint, quant_config


def maybe_set_tp_grain_size(quant_config, ds_init_inf_kwargs):
    tp_grain_size = 64
    # Need to check if this attr is available. Old DeepSpeep does not have it.
    assert "tp_grain_size" in dir(
        deepspeed.inference.config.DeepSpeedTPConfig()
    ), "Old DeepSpeed version detected. Please update to the recommended version."
    if quant_config is not None:
        assert "group_size" in quant_config
        group_size = quant_config["group_size"]
        if group_size > 0:
            tp_grain_size = group_size
        ds_init_inf_kwargs.update(
            {
                "tensor_parallel": deepspeed.inference.config.DeepSpeedTPConfig(
                    tp_grain_size=tp_grain_size
                )
            }
        )
    return tp_grain_size


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
        add_special_tokens=True,
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
        self.load_dtype = load_dtype

        model_type = next(
            (x for x in MODEL_CLASSES.keys() if x in model_id.lower()), "auto"
        )
        model_class = MODEL_CLASSES[model_type]

        self.tokenizer = model_class[1].from_pretrained(
            model_id, trust_remote_code=True
        )
        if model_type == "chatglm":
            # chatglm modeling is from remote hub and its torch_dtype in config.json need to be overrided
            self.config = AutoConfig.from_pretrained(
                model_id if config is None else config,
                torchscript=with_jit,
                trust_remote_code=True,
                torch_dtype=load_dtype,
            )
        else:
            self.config = AutoConfig.from_pretrained(
                model_id if config is None else config,
                torchscript=with_jit,
                trust_remote_code=True,
            )
        if re.search("gptbigcode", self.config.architectures[0], re.IGNORECASE):
            model_type = "gptbigcode"

        # For now, Falcon, baichuan and gptbigcode have accuracy issue with from_config with deepspeed meta device load.
        # TODO: we will change the scope once deepspeed providing the support
        if world_size == 1 or model_type in [
            "falcon",
            "baichuan",
            "baichuan2",
            "gptbigcode",
            "qwen3moe",
            "qwen3",
            "qwen",
            "yuan",
            "jamba",
        ]:
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
                    self.model = model_class[0](config=self.config)
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
                # use absolute path here to avoid path error in deepspeed
                model_name_or_path = os.path.abspath(model_name_or_path)
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
            glob_pattern = "*.[bp][it][n]"
            if (
                re.search("deepseek-v2", model_name_or_path, re.IGNORECASE)
                or re.search("deepseek-v3", model_name_or_path, re.IGNORECASE)
                or re.search("deepseek-r1", model_name_or_path, re.IGNORECASE)
            ):
                glob_pattern = "*.[sbp][ait][fn][e][t][e][n][s][o][r][s]"
            # extensions: .bin | .pt
            # creates a list of paths from all downloaded files in cache dir
            file_list = [
                str(entry)
                for entry in Path(cached_repo_dir).rglob(glob_pattern)
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
        low_precision_checkpoint, quant_config = get_low_precision_checkpoint(
            args, self.config
        )
        tp_grain_size = maybe_set_tp_grain_size(quant_config, ds_init_inf_kwargs)
        if not args.ipex_weight_only_quantization or low_precision_checkpoint is None:
            ds_init_inf_kwargs.update({"checkpoint": checkpoints_json})
        self.model = deepspeed.init_inference(
            self.model,
            mp_size=tp_number,
            base_dir=repo_root,
            dtype=infer_dtype,
            **ds_init_inf_kwargs,
        )

        self.model = self.model.module

        if self._with_ipex:
            ipex_woq_enabled = args.ipex_weight_only_quantization
            low_precision_ckpt = None
            if ipex_woq_enabled:
                from intel_extension_for_pytorch.quantization import (
                    WoqWeightDtype,
                    WoqWeightQScheme,
                )

                if args.weight_dtype == "INT8":
                    weight_dtype = WoqWeightDtype.INT8
                elif args.weight_dtype == "INT4":
                    weight_dtype = WoqWeightDtype.INT4
                elif args.weight_dtype == "NF4":
                    weight_dtype = WoqWeightDtype.NF4
                else:
                    assert args.weight_dtype == "FP8"
                    weight_dtype = WoqWeightDtype.FP8

                if args.lowp_mode == "INT8":
                    lowp_mode = ipex.quantization.WoqLowpMode.INT8
                elif args.lowp_mode == "FP32":
                    lowp_mode = ipex.quantization.WoqLowpMode.NONE
                elif args.lowp_mode == "FP16":
                    lowp_mode = ipex.quantization.WoqLowpMode.FP16
                elif args.lowp_mode == "BF16":
                    lowp_mode = ipex.quantization.WoqLowpMode.BF16
                else:  # AUTO
                    if weight_dtype == WoqWeightDtype.INT4 or (
                        low_precision_checkpoint is not None
                        and quant_config["bits"] == 4
                    ):
                        lowp_mode = ipex.quantization.WoqLowpMode.INT8
                    else:
                        lowp_mode = ipex.quantization.WoqLowpMode.BF16

                act_quant_mode_dict = {
                    "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
                    "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                    "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
                    "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
                    "PER_TENSOR_SYM": ipex.quantization.WoqActQuantMode.PER_TENSOR_SYM,
                    "PER_IC_BLOCK_SYM": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK_SYM,
                    "PER_BATCH_SYM": ipex.quantization.WoqActQuantMode.PER_BATCH_SYM,
                    "PER_BATCH_IC_BLOCK_SYM": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
                }
                weight_qscheme = (
                    WoqWeightQScheme.SYMMETRIC
                    if args.woq_sym_quant_weight
                    else WoqWeightQScheme.UNDEFINED
                )
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    weight_dtype=weight_dtype,
                    lowp_mode=lowp_mode,
                    act_quant_mode=act_quant_mode_dict[args.act_quant_mode],
                    group_size=args.group_size,
                    weight_qscheme=weight_qscheme,
                )
                model = self.model
                if low_precision_checkpoint is not None:
                    rank = local_rank
                    assert "quant_method" in quant_config
                    assert "desc_act" in quant_config
                    quant_method = quant_config["quant_method"]
                    desc_act = quant_config["desc_act"]
                    bits = quant_config["bits"]
                    if (
                        world_size > 1
                        and desc_act
                        and lowp_mode == ipex.quantization.WoqLowpMode.INT8
                    ):
                        raise AssertionError(
                            "Lowp-mode INT8 is not supported for TP with desc_act = True"
                        )
                    low_precision_ckpt = shard_low_precision_checkpoint(
                        low_precision_checkpoint,
                        model.config,
                        rank,
                        world_size,
                        quant_method,
                        tp_grain_size,
                        desc_act,
                        bits,
                    )
                    low_precision_ckpt = (low_precision_ckpt, quant_config)
            self.model = ipex.llm.optimize(
                self.model.eval(),
                dtype=infer_dtype,
                quantization_config=qconfig if ipex_woq_enabled else None,
                inplace=True,
                deployment_mode=False,
                cache_weight_for_large_batch=args.cache_weight_for_large_batch,
                low_precision_checkpoint=low_precision_ckpt,
            )

        self.base_model = self.model

        self.num_beams = 1 if with_greedy else 4
        self.iter = 0
        self.is_t5 = re.search(
            "t5", self.base_model.config.architectures[0], re.IGNORECASE
        )

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
                    (
                        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                        torch.zeros([1, num_heads, 1, head_dim]).contiguous(),
                        torch.zeros([1, num_heads, 1, head_dim]).contiguous(),
                        beam_idx_tmp,
                        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                        self.base_model.decoder.block[i]
                        .layer[1]
                        .EncDecAttention.k(last_hidden_state)
                        .view(
                            int(input_bs),
                            -1,
                            self.base_model.decoder.block[i]
                            .layer[1]
                            .EncDecAttention.n_heads,
                            self.base_model.decoder.block[i]
                            .layer[1]
                            .EncDecAttention.key_value_proj_dim,
                        )
                        .transpose(0, 1)
                        .contiguous(),
                        self.base_model.decoder.block[i]
                        .layer[1]
                        .EncDecAttention.v(last_hidden_state)
                        .view(
                            int(input_bs),
                            -1,
                            self.base_model.decoder.block[i]
                            .layer[1]
                            .EncDecAttention.n_heads,
                            self.base_model.decoder.block[i]
                            .layer[1]
                            .EncDecAttention.key_value_proj_dim,
                        )
                        .transpose(0, 1)
                        .contiguous(),
                        beam_idx_tmp,
                    )
                    if self.is_t5
                    else (
                        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                        torch.zeros([1, num_heads, 1, head_dim]).contiguous(),
                        torch.zeros([1, num_heads, 1, head_dim]).contiguous(),
                        beam_idx_tmp,
                    )
                )
                for i in range(num_hidden_layers)
            ]
        )
        if re.search("yuan", self.config.architectures[0], re.IGNORECASE):
            past_key_values = tuple(
                [
                    (
                        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        torch.zeros(1, 4, dtype=torch.long),
                        torch.zeros(1, 1, 2, hidden_size),
                    )
                    for i in range(num_hidden_layers)
                ]
            )
        if re.search("jamba", self.config.architectures[0], re.IGNORECASE):
            intermediate_size = self.config.mamba_expand * self.config.hidden_size
            conv_kernel_size = self.config.mamba_d_conv
            ssm_state_size = self.config.mamba_d_state
            past_key_values = tuple(
                [
                    (
                        (
                            torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                            torch.zeros([1, 1, 1, 1]).contiguous(),
                            torch.zeros([1, 1, 1, 1]).contiguous(),
                            torch.zeros(1, 4, dtype=torch.long),
                        )
                        if i % self.config.attn_layer_period
                        == self.config.attn_layer_offset
                        else (
                            torch.zeros(
                                input_bs,
                                intermediate_size,
                                ssm_state_size,
                                dtype=self.load_dtype,
                            ).contiguous(),
                            torch.zeros(
                                input_bs,
                                intermediate_size,
                                conv_kernel_size,
                                dtype=self.load_dtype,
                            ).contiguous(),
                            torch.tensor(False).contiguous(),
                        )
                    )
                    for i in range(self.config.num_hidden_layers)
                ]
            )
        if re.search("deepseek", self.config.architectures[0], re.IGNORECASE):
            past_key_values = tuple(
                [
                    (
                        (
                            torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                            torch.zeros([1, 1, 1, 1]).contiguous(),
                            torch.zeros(1, 4, dtype=torch.long),
                        )
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
            inputs = inputs["input_ids"]
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
            model_kwargs = (
                self.base_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs,
                    model_kwargs,
                    "input_ids",
                    transformers.generation.configuration_utils.GenerationConfig(),
                )
            )
            (
                inputs,
                example_inputs,
            ) = self.base_model._expand_inputs_for_generation(
                input_ids=inputs,
                expand_size=self.num_beams,
                is_encoder_decoder=True,
                **model_kwargs,
            )
            past_key_values = self._get_past_key_values(
                input_bs, example_inputs["encoder_outputs"]["last_hidden_state"]
            )
            if self.num_beams == 1:
                decoder_input_ids = self.base_model._shift_right(labels["input_ids"])
                _labels = labels["input_ids"]
            else:
                decoder_input_ids = self.base_model._shift_right(
                    labels["input_ids"].repeat_interleave(self.num_beams, dim=0)
                )
                _labels = labels["input_ids"].repeat_interleave(self.num_beams, dim=0)
            example_dict = {
                "decoder_input_ids": decoder_input_ids,
                "encoder_outputs": (
                    example_inputs["encoder_outputs"]["last_hidden_state"],
                ),
                "labels": _labels,
            }
        else:
            past_key_values = self._get_past_key_values(input_bs)
            example_dict = {"input_ids": inputs}
        model_inputs = self.base_model.prepare_inputs_for_generation(
            inputs, attention_mask=attention_mask_batched
        )
        has_position_ids = model_inputs.get("position_ids", None) is not None
        if self._with_ipex:
            example_dict["attention_mask"] = attention_mask_batched
            example_dict["past_key_values"] = past_key_values
            example_dict["return_dict"] = torch.tensor(False)
            if has_position_ids:
                example_dict["position_ids"] = position_ids_batched
        if "return_last_logit" in model_inputs:
            example_dict["return_last_logit"] = torch.tensor(True)
        if "output_router_logits" in model_inputs:
            example_dict["output_router_logits"] = torch.tensor(
                model_inputs["output_router_logits"]
            )
        if self.config.architectures[0] == "JambaForCausalLM":
            example_dict["num_logits_to_keep"] = torch.tensor(0)

        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=True if args.quant_with_amp or self._dtype == "bfloat16" else False,
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
        if re.search("yuan", self.base_model.config.architectures[0], re.IGNORECASE):
            return 1024
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

    def greedy_until(self, requests):
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        warn_stop_seq = False
        for context, request_args in tqdm(re_ord.get_reordered()):
            until = request_args["until"]
            if isinstance(until, str):
                until = [until]

            if until:
                try:
                    (primary_until,) = self.tok_encode(until[0])
                except ValueError:
                    if not warn_stop_seq:
                        print(
                            "Warning: a primary stop sequence is multi-token! Will default to EOS token for"
                            " this tokenizer. Consider using `hf-causal-experimental` for multi-token stop"
                            " sequence support for the time being."
                        )
                        warn_stop_seq = True
                    primary_until = self.eot_token_id
            else:
                primary_until = None
            if re.search(
                "yuan", self.base_model.config.architectures[0], re.IGNORECASE
            ):
                context = "详细分析并求解以下数学问题。\n" + context.replace(
                    "问题: ", ""
                ).replace("\n逐步解答:", "<sep>")
            context_enc = torch.tensor(
                [self.tok_encode(context)[self.max_gen_toks - self.max_length :]]
            ).to(self.device)

            max_gen_tokens = min(
                self.max_gen_toks, request_args.get("max_length", self.max_gen_toks)
            )

            with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                enabled=(
                    True if args.quant_with_amp or self._dtype == "bfloat16" else False
                ),
            ):
                if self._with_jit and self.iter == 0:
                    if self._dtype not in ["int8", "int4", "nf4"]:
                        if re.search(
                            "yuan",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        ):
                            input_bs = context_enc.shape[0] * self.num_beams
                            attention_mask = torch.ones(len(context_enc[0]))
                            position_ids = torch.arange(len(context_enc[0]))
                            example_dict = {
                                "input_ids": context_enc[:, -1:],
                                "attention_mask": attention_mask.unsqueeze(0)[:, -1:],
                                "position_ids": position_ids.unsqueeze(0)[:, -1:],
                                "past_key_values": self._get_past_key_values(input_bs),
                            }
                            model = torch.jit.trace(
                                self.model.eval(),
                                example_kwarg_inputs=example_dict,
                                strict=False,
                                check_trace=False,
                            )
                            model = torch.jit.freeze(model.eval())
                            example_dict = {
                                "input_ids": example_dict["input_ids"].repeat(
                                    input_bs, 1
                                ),
                                "attention_mask": example_dict["attention_mask"].repeat(
                                    input_bs, 1
                                ),
                                "position_ids": example_dict["position_ids"].repeat(
                                    input_bs, 1
                                ),
                            }
                            first_token_model = torch.jit.trace(
                                self.model.eval(),
                                example_kwarg_inputs=example_dict,
                                strict=False,
                                check_trace=False,
                            )
                            first_token_model = torch.jit.freeze(
                                first_token_model.eval()
                            )
                    else:
                        model = torch.jit.load(args.quantized_model_path)
                        model = torch.jit.freeze(model.eval())
                        if re.search(
                            "yuan",
                            self.base_model.config.architectures[0],
                            re.IGNORECASE,
                        ):
                            first_token_model = torch.jit.load(
                                args.quantized_model_path + "2"
                            )
                            first_token_model = torch.jit.freeze(
                                first_token_model.eval()
                            )
                    if re.search(
                        "yuan", self.base_model.config.architectures[0], re.IGNORECASE
                    ):
                        ipex._set_optimized_model_for_generation(
                            self.model,
                            optimized_model=model,
                            first_token_optimized_model=first_token_model,
                        )
                    else:
                        ipex._set_optimized_model_for_generation(
                            self.model, optimized_model=model
                        )
                    self.iter = self.iter + 1
                cont = self._model_generate(
                    context_enc, context_enc.shape[1] + max_gen_tokens, primary_until
                )

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1] :])
            if re.search(
                "yuan", self.base_model.config.architectures[0], re.IGNORECASE
            ):
                s = s.replace("\n", "").split("<eod>")[0]

            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)

            res.append(s)

        return re_ord.get_original(res)


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
                if (
                    length >= 1
                    and target_tokens[length - 1].item()
                    == self.tokenizer.encode(
                        self.tokenizer.eos_token, add_special_tokens=False
                    )[0]
                ):
                    length = length - 1

                log_softmax = log_softmax[:length]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_text = self.tokenizer.decode(
                    target_tokens, skip_special_tokens=True
                )
                greedy_text = self.tokenizer.decode(
                    greedy_tokens, skip_special_tokens=True
                )
                max_text_equal = greedy_text == target_text
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal or max_text_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return results


@register_model("test")
class LMMS(lmms):
    def __init__(
        self,
        pretrained: str,
        device: Optional[str] = "cpu",
        with_ipex=True,
        with_jit=True,
        with_greedy=False,
        batch_size=1,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        tp_number=1,
        config=None,
        add_special_tokens=True,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = int(batch_size)
        self._with_jit = with_jit
        self._with_ipex = with_ipex
        self._with_greedy = with_greedy
        self._tp_number = tp_number
        self._dtype = dtype
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
        self.amp_dtype = (
            torch.bfloat16
            if args.quant_with_amp or self._dtype == "bfloat16"
            else torch.float32
        )
        if re.search("llava", pretrained, re.IGNORECASE):
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    pretrained, None, get_model_name_from_path(pretrained)
                )
            )
            model_name = get_model_name_from_path(pretrained)
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"
            self.conv_template = conv_mode
        elif re.search("git", pretrained, re.IGNORECASE):
            model_class = MODEL_CLASSES["git"]
            self._image_processor = model_class[1].from_pretrained(
                pretrained, trust_remote_code=True
            )
            self._tokenizer = self._image_processor.tokenizer
            self._config = AutoConfig.from_pretrained(
                pretrained if config is None else config,
                torchscript=with_jit,
                trust_remote_code=True,
            )
            self._model = model_class[0].from_pretrained(
                pretrained,
                low_cpu_mem_usage=True,
                config=self.config,
                torch_dtype=load_dtype,
                trust_remote_code=True,
            )
        elif re.search("llama", pretrained, re.IGNORECASE):
            model_class = MODEL_CLASSES["mllama"]
            self._config = AutoConfig.from_pretrained(
                pretrained if config is None else config,
                torchscript=with_jit,
                trust_remote_code=True,
            )
            self._model = model_class[0].from_pretrained(
                pretrained,
                low_cpu_mem_usage=True,
                config=self.config,
                torch_dtype=load_dtype,
                trust_remote_code=True,
            )
            self._image_processor = model_class[1].from_pretrained(pretrained)
            self._tokenizer = self._image_processor.tokenizer
        self._config = self._model.config
        self._config.torchscript = self._with_jit
        self._model.eval()

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
            checkpoint_files = get_checkpoint_files(pretrained)
            if local_rank == 0:
                # model.config.model_type.upper()
                data = {
                    "type": "BLOOM",
                    "checkpoints": checkpoint_files,
                    "version": 1.0,
                }
                json.dump(data, open(checkpoints_json, "w"))

        repo_root = get_repo_root(pretrained)
        write_checkpoints_json()
        dist.barrier()
        low_precision_checkpoint, quant_config = get_low_precision_checkpoint(
            args, self.config
        )
        tp_grain_size = maybe_set_tp_grain_size(quant_config, ds_init_inf_kwargs)
        if not args.ipex_weight_only_quantization or low_precision_checkpoint is None:
            ds_init_inf_kwargs.update({"checkpoint": checkpoints_json})
        self._model = deepspeed.init_inference(
            self._model,
            mp_size=tp_number,
            base_dir=repo_root,
            dtype=infer_dtype,
            **ds_init_inf_kwargs,
        )

        self._model = self._model.module

        if self._with_ipex:
            ipex_woq_enabled = args.ipex_weight_only_quantization
            low_precision_ckpt = None
            if ipex_woq_enabled:
                from intel_extension_for_pytorch.quantization import (
                    WoqWeightDtype,
                    WoqWeightQScheme,
                )

                weight_dtype = (
                    WoqWeightDtype.INT4
                    if args.weight_dtype == "INT4"
                    else WoqWeightDtype.INT8
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
                    if weight_dtype == WoqWeightDtype.INT4 or (
                        low_precision_checkpoint is not None
                        and quant_config["bits"] == 4
                    ):
                        lowp_mode = ipex.quantization.WoqLowpMode.INT8
                    else:
                        lowp_mode = ipex.quantization.WoqLowpMode.BF16

                act_quant_mode_dict = {
                    "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
                    "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                    "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
                    "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
                    "PER_TENSOR_SYM": ipex.quantization.WoqActQuantMode.PER_TENSOR_SYM,
                    "PER_IC_BLOCK_SYM": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK_SYM,
                    "PER_BATCH_SYM": ipex.quantization.WoqActQuantMode.PER_BATCH_SYM,
                    "PER_BATCH_IC_BLOCK_SYM": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
                }
                weight_qscheme = (
                    WoqWeightQScheme.SYMMETRIC
                    if args.woq_sym_quant_weight
                    else WoqWeightQScheme.UNDEFINED
                )
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    weight_dtype=weight_dtype,
                    lowp_mode=lowp_mode,
                    act_quant_mode=act_quant_mode_dict[args.act_quant_mode],
                    group_size=args.group_size,
                    weight_qscheme=weight_qscheme,
                )
                model = self.model
                if low_precision_checkpoint is not None:
                    rank = local_rank
                    assert "quant_method" in quant_config
                    assert "desc_act" in quant_config
                    quant_method = quant_config["quant_method"]
                    desc_act = quant_config["desc_act"]
                    bits = quant_config["bits"]
                    if (
                        world_size > 1
                        and desc_act
                        and lowp_mode == ipex.quantization.WoqLowpMode.INT8
                    ):
                        raise AssertionError(
                            "Lowp-mode INT8 is not supported for TP with desc_act = True"
                        )
                    low_precision_ckpt = shard_low_precision_checkpoint(
                        low_precision_checkpoint,
                        model.config,
                        rank,
                        world_size,
                        quant_method,
                        tp_grain_size,
                        desc_act,
                        bits,
                    )
                    low_precision_ckpt = (low_precision_ckpt, quant_config)
            self._model = ipex.llm.optimize(
                self._model.eval(),
                dtype=infer_dtype,
                quantization_config=qconfig if ipex_woq_enabled else None,
                inplace=True,
                deployment_mode=False,
                cache_weight_for_large_batch=args.cache_weight_for_large_batch,
                low_precision_checkpoint=low_precision_ckpt,
            )

        self._base_model = self._model

        self.iter = 0
        self.num_beams = 1 if with_greedy else 4
        if self._with_jit:
            input_ids = torch.ones(1).to(torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            if self._model.config.architectures[0] == "MllamaForConditionalGeneration":
                head_dim = self._model.config.text_config.hidden_size // (
                    self._model.config.text_config.num_hidden_layers
                    - len(self._model.config.text_config.cross_attention_layers)
                )
                past_key_values = tuple(
                    [
                        (
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                torch.zeros(1, 4, dtype=torch.long),
                            )
                            if i
                            not in self._model.config.text_config.cross_attention_layers
                            else (
                                torch.zeros([1, 1, 1, head_dim]).contiguous(),
                                torch.zeros([1, 1, 1, head_dim]).contiguous(),
                            )
                        )
                        for i in range(self._model.config.text_config.num_hidden_layers)
                    ]
                )
            else:
                past_key_values = tuple(
                    [
                        (
                            (
                                torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                torch.zeros([1, 1, 1, 1]).contiguous(),
                                torch.zeros(1, 4, dtype=torch.long),
                            )
                        )
                        for i in range(self.model.config.num_hidden_layers)
                    ]
                )
            sample_inputs = {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            if re.search("llava", pretrained, re.IGNORECASE):
                sample_inputs["inputs_embeds"] = torch.zeros(batch_size, 1, 4096).to(
                    self.amp_dtype
                )
            elif re.search("git", pretrained, re.IGNORECASE):
                sample_inputs["input_ids"] = torch.ones(batch_size, 1).to(torch.long)
                sample_inputs["attention_mask"] = torch.ones(batch_size, 1)
                sample_inputs["pixel_values"] = torch.zeros(batch_size, 3, 224, 224)
                num_head = self.model.git.encoder.layer[
                    0
                ].attention.self.num_attention_heads
                head_dim = int(
                    self.model.git.encoder.layer[0].attention.self.hidden_size
                    / num_head
                )
                past_key_values = tuple(
                    [
                        (
                            torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                            torch.zeros(
                                [batch_size, num_head, 1, head_dim]
                            ).contiguous(),
                            torch.zeros(
                                [batch_size, num_head, 1, head_dim]
                            ).contiguous(),
                            torch.zeros(1, 4, dtype=torch.long),
                        )
                        for i in range(self.model.config.num_hidden_layers)
                    ]
                )
                sample_inputs["past_key_values"] = past_key_values
            elif (
                self._model.config.architectures[0] == "MllamaForConditionalGeneration"
            ):
                input_ids = torch.ones(32).to(torch.long).unsqueeze(0)
                position_ids = torch.arange(len(input_ids)).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids).unsqueeze(0)
                cross_attention_mask = torch.ones(1, 32, 1, 4)
                pixel_values = torch.rand(
                    1,
                    1,
                    4,
                    3,
                    self._model.config.vision_config.image_size,
                    self._model.config.vision_config.image_size,
                )
                aspect_ratio_mask = torch.tensor([[[1, 1, 1, 1]]])
                aspect_ratio_ids = torch.tensor([[6]])
                sample_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "cross_attention_mask": cross_attention_mask,
                    "pixel_values": pixel_values,
                    "aspect_ratio_mask": aspect_ratio_mask,
                    "aspect_ratio_ids": aspect_ratio_ids,
                }
                sample_inputs["cross_attention_mask"] = cross_attention_mask
            with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True if self.amp_dtype == torch.bfloat16 else False,
            ):
                if self._dtype != "int8":
                    if (
                        self._model.config.architectures[0]
                        == "MllamaForConditionalGeneration"
                    ):
                        traced_model_first = torch.jit.trace(
                            self._model.eval(),
                            example_kwarg_inputs=sample_inputs,
                            strict=False,
                            check_trace=False,
                        )
                        traced_model_first = torch.jit.freeze(traced_model_first.eval())
                        traced_model_first(**sample_inputs)
                        traced_model_first(**sample_inputs)
                        sample_inputs.pop("pixel_values", None)
                        sample_inputs.pop("aspect_ratio_mask", None)
                        sample_inputs.pop("aspect_ratio_ids", None)
                    traced_model = torch.jit.trace(
                        self._model.eval(),
                        example_kwarg_inputs=sample_inputs,
                        strict=False,
                        check_trace=False,
                    )
                    traced_model = torch.jit.freeze(traced_model)
                else:
                    if (
                        self._model.config.architectures[0]
                        == "MllamaForConditionalGeneration"
                    ):
                        traced_model_first = torch.jit.load(
                            args.quantized_model_path + "2"
                        )
                        traced_model_first = torch.jit.freeze(traced_model_first.eval())
                        traced_model_first(**sample_inputs)
                        traced_model_first(**sample_inputs)
                        sample_inputs.pop("pixel_values", None)
                        sample_inputs.pop("aspect_ratio_mask", None)
                        sample_inputs.pop("aspect_ratio_ids", None)
                    traced_model = torch.jit.load(args.quantized_model_path)
                    traced_model = torch.jit.freeze(traced_model)

                traced_model(**sample_inputs)
                traced_model(**sample_inputs)
            if self._model.config.architectures[0] == "MllamaForConditionalGeneration":
                ipex._set_optimized_model_for_generation(
                    self._model,
                    optimized_model=traced_model,
                    first_token_optimized_model=traced_model_first,
                )
            else:
                ipex._set_optimized_model_for_generation(
                    self._model, optimized_model=traced_model
                )

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [
                        _image.to(dtype=torch.float16, device=self.device)
                        for _image in image
                    ]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0]

            if (
                image is not None
                and len(image) != 0
                and DEFAULT_IMAGE_TOKEN not in prompts_input
            ):
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it.
                    In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts[0]

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            contxt_id = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids, labels=labels, images=image, use_cache=True
                )
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[
                :, contxt_id.shape[1] : input_ids.shape[1]
            ]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = lmms_utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if re.search("llava", self.model.config.architectures[0], re.IGNORECASE):
                # Set default values for until and max_new_tokens
                until = [self.tok_decode(self.eot_token_id)]

                # Update values from gen_kwargs if present
                if "until" in gen_kwargs:
                    until = gen_kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                        )

                if (
                    "image_aspect_ratio" in gen_kwargs.keys()
                    and "image_aspect_ratio" not in self._config.__dict__
                ):
                    # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                    self._config.image_aspect_ratio = gen_kwargs.pop(
                        "image_aspect_ratio"
                    )
                # encode, pad, and truncate contexts for this batch
                if visuals:
                    image_tensor = process_images(
                        visuals, self._image_processor, self._config
                    )
                else:
                    image_tensor = None

                # prompts_input = contexts[0]

                question_input = []

                for visual, context in zip(visuals, contexts):
                    if (
                        image_tensor is not None
                        and len(image_tensor) != 0
                        and DEFAULT_IMAGE_TOKEN not in context
                    ):
                        """
                        Three senarios:
                        1. No image, and there for, no image token should be added.
                        2. image token is already specified in the context, so we don't need to add it.
                        3. image token is not specified in the context and there is image inputs,
                            so we need to add it. In this case, we add the image token at the beginning
                            of the context and add a new line.
                        """
                        image_tokens = (
                            [DEFAULT_IMAGE_TOKEN] * len(visual)
                            if isinstance(visual, list)
                            else [DEFAULT_IMAGE_TOKEN]
                        )
                        image_tokens = " ".join(image_tokens)
                        question = image_tokens + "\n" + context
                    else:
                        question = context

                    conv = conv_templates[self.conv_template].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

                # The above for loop has bugs. When there is no visuals, e.g. pure text,
                # there will be no for loop execute resulting in an empty question_input (because no visuals)
                # Scenario 1 won't even be execute
                if len(visuals) == 0:
                    for context in contexts:
                        question = context
                        conv = conv_templates[self.conv_template].copy()
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)

                # preconfigure gen_kwargs with defaults
                gen_kwargs["image_sizes"] = [
                    visuals[idx].size for idx in range(len(visuals))
                ]
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                input_ids_list = [
                    tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    for prompt in question_input
                ]
                pad_token_ids = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                )
                input_ids = self.pad_sequence(
                    input_ids_list, batch_first=True, padding_value=pad_token_ids
                ).to(get_accelerator().current_device_name())
                attention_masks = input_ids.ne(pad_token_ids).to(
                    get_accelerator().current_device_name()
                )
                input_dict = {
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "pad_token_id": pad_token_ids,
                    "images": image_tensor.to(self.amp_dtype),
                    "do_sample": True if gen_kwargs["temperature"] > 0 else False,
                    "temperature": gen_kwargs["temperature"],
                    "top_p": gen_kwargs["top_p"],
                    "num_beams": gen_kwargs["num_beams"],
                    "max_new_tokens": gen_kwargs["max_new_tokens"],
                }
            elif re.search("git", self.model.config.architectures[0], re.IGNORECASE):
                input_ids = self._image_processor(
                    images=visuals, return_tensors="pt"
                ).pixel_values
                gen_kwargs.pop("until", None)
                input_dict = {
                    "pixel_values": input_ids.to(self.amp_dtype),
                    **gen_kwargs,
                }
            elif re.search("mllama", self.model.config.architectures[0], re.IGNORECASE):
                prompts = [
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|><|image|>"
                    + context
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                    for context in contexts
                ]
                inputs = self._image_processor(
                    text=prompts, images=visuals, return_tensors="pt"
                )
                input_ids = inputs["input_ids"]
                gen_kwargs.pop("until", None)
                input_dict = {
                    **inputs,
                    **gen_kwargs,
                }
            try:
                with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                    enabled=True if self.amp_dtype == torch.bfloat16 else False,
                ):
                    cont = self.model.generate(**input_dict)
                    text_outputs = self.tokenizer.batch_decode(
                        (
                            cont[:, input_ids.shape[1] :]
                            if re.search(
                                "llava",
                                self.model.config.architectures[0],
                                re.IGNORECASE,
                            )
                            or re.search(
                                "mllama",
                                self.model.config.architectures[0],
                                re.IGNORECASE,
                            )
                            else cont
                        ),
                        skip_special_tokens=True,
                    )
            except Exception as e:
                print(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]
            res.extend(text_outputs)
            # self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        res = re_ords.get_original(res)

        pbar.close()
        return res


class LibriSpeech:
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
        add_special_tokens=True,
    ):
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
        self.amp_dtype = (
            torch.bfloat16
            if args.quant_with_amp or self._dtype == "bfloat16"
            else torch.float32
        )
        model_type = next(
            (x for x in MODEL_CLASSES.keys() if x in model_id.lower()), "auto"
        )
        model_class = MODEL_CLASSES[model_type]

        self.tokenizer = model_class[1].from_pretrained(
            model_id, trust_remote_code=True
        )
        self.config = AutoConfig.from_pretrained(
            model_id if config is None else config,
            torchscript=with_jit,
            trust_remote_code=True,
        )

        # For now, Falcon, baichuan and gptbigcode have accuracy issue with from_config with deepspeed meta device load.
        # TODO: we will change the scope once deepspeed providing the support
        if world_size == 1 or model_type in [
            "whisper",
        ]:
            self.model = model_class[0].from_pretrained(
                model_id,
                config=self.config,
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
            if os.path.exists(model_name_or_path):
                # local path
                # use absolute path here to avoid path error in deepspeed
                model_name_or_path = os.path.abspath(model_name_or_path)
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
        low_precision_checkpoint, quant_config = get_low_precision_checkpoint(
            args, self.config
        )
        tp_grain_size = maybe_set_tp_grain_size(quant_config, ds_init_inf_kwargs)
        if not args.ipex_weight_only_quantization or low_precision_checkpoint is None:
            ds_init_inf_kwargs.update({"checkpoint": checkpoints_json})
        self.model = deepspeed.init_inference(
            self.model,
            mp_size=tp_number,
            base_dir=repo_root,
            dtype=infer_dtype,
            **ds_init_inf_kwargs,
        )

        self.model = self.model.module

        if self._with_ipex:
            ipex_woq_enabled = args.ipex_weight_only_quantization
            low_precision_ckpt = None
            if ipex_woq_enabled:
                from intel_extension_for_pytorch.quantization import (
                    WoqWeightDtype,
                    WoqWeightQScheme,
                )

                if args.weight_dtype == "INT8":
                    weight_dtype = WoqWeightDtype.INT8
                elif args.weight_dtype == "INT4":
                    weight_dtype = WoqWeightDtype.INT4
                elif args.weight_dtype == "NF4":
                    weight_dtype = WoqWeightDtype.NF4
                else:
                    assert args.weight_dtype == "FP8"
                    weight_dtype = WoqWeightDtype.FP8

                if args.lowp_mode == "INT8":
                    lowp_mode = ipex.quantization.WoqLowpMode.INT8
                elif args.lowp_mode == "FP32":
                    lowp_mode = ipex.quantization.WoqLowpMode.NONE
                elif args.lowp_mode == "FP16":
                    lowp_mode = ipex.quantization.WoqLowpMode.FP16
                elif args.lowp_mode == "BF16":
                    lowp_mode = ipex.quantization.WoqLowpMode.BF16
                else:  # AUTO
                    if weight_dtype == WoqWeightDtype.INT4 or (
                        low_precision_checkpoint is not None
                        and quant_config["bits"] == 4
                    ):
                        lowp_mode = ipex.quantization.WoqLowpMode.INT8
                    else:
                        lowp_mode = ipex.quantization.WoqLowpMode.BF16

                act_quant_mode_dict = {
                    "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
                    "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                    "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
                    "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
                    "PER_TENSOR_SYM": ipex.quantization.WoqActQuantMode.PER_TENSOR_SYM,
                    "PER_IC_BLOCK_SYM": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK_SYM,
                    "PER_BATCH_SYM": ipex.quantization.WoqActQuantMode.PER_BATCH_SYM,
                    "PER_BATCH_IC_BLOCK_SYM": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK_SYM,
                }
                weight_qscheme = (
                    WoqWeightQScheme.SYMMETRIC
                    if args.woq_sym_quant_weight
                    else WoqWeightQScheme.UNDEFINED
                )
                qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                    weight_dtype=weight_dtype,
                    lowp_mode=lowp_mode,
                    act_quant_mode=act_quant_mode_dict[args.act_quant_mode],
                    group_size=args.group_size,
                    weight_qscheme=weight_qscheme,
                )
                model = self.model
                if low_precision_checkpoint is not None:
                    rank = local_rank
                    assert "quant_method" in quant_config
                    assert "desc_act" in quant_config
                    quant_method = quant_config["quant_method"]
                    desc_act = quant_config["desc_act"]
                    bits = quant_config["bits"]
                    if (
                        world_size > 1
                        and desc_act
                        and lowp_mode == ipex.quantization.WoqLowpMode.INT8
                    ):
                        raise AssertionError(
                            "Lowp-mode INT8 is not supported for TP with desc_act = True"
                        )
                    low_precision_ckpt = shard_low_precision_checkpoint(
                        low_precision_checkpoint,
                        model.config,
                        rank,
                        world_size,
                        quant_method,
                        tp_grain_size,
                        desc_act,
                        bits,
                    )
                    low_precision_ckpt = (low_precision_ckpt, quant_config)
            self.model = ipex.llm.optimize(
                self.model.eval(),
                dtype=infer_dtype,
                quantization_config=qconfig if ipex_woq_enabled else None,
                inplace=True,
                deployment_mode=False,
                cache_weight_for_large_batch=args.cache_weight_for_large_batch,
                low_precision_checkpoint=low_precision_ckpt,
            )

        self.base_model = self.model

        self.num_beams = 1 if with_greedy else 4
        self.iter = 0

        if self._with_jit:
            past_key_values = tuple(
                [
                    (
                        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        torch.zeros([1, 1, 1, 1]).contiguous(),
                        torch.zeros(1, 4, dtype=torch.long),
                        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                        torch.zeros(
                            [
                                1,
                                32,
                                self.model.model.decoder.layers[
                                    i
                                ].encoder_attn.num_heads,
                                self.model.model.decoder.layers[
                                    i
                                ].encoder_attn.head_dim,
                            ],
                            dtype=self.amp_dtype,
                        ).contiguous(),
                        torch.zeros(
                            [
                                1,
                                32,
                                self.model.model.decoder.layers[
                                    i
                                ].encoder_attn.num_heads,
                                self.model.model.decoder.layers[
                                    i
                                ].encoder_attn.head_dim,
                            ],
                            dtype=self.amp_dtype,
                        ).contiguous(),
                        torch.zeros(1, 4, dtype=torch.long),
                    )
                    for i in range(self.config.num_hidden_layers)
                ]
            )
            last_hidden_state = torch.rand([1, 32, 1280]).to(self.amp_dtype)
            sample_inputs = {
                "decoder_input_ids": torch.ones(4).to(torch.long).unsqueeze(0),
                "past_key_values": past_key_values,
                "encoder_outputs": (last_hidden_state,),
            }
            with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True if self.amp_dtype == torch.bfloat16 else False,
            ):
                if self._dtype != "int8":
                    traced_model = torch.jit.trace(
                        self.model.eval(),
                        example_kwarg_inputs=sample_inputs,
                        strict=False,
                        check_trace=False,
                    )
                    traced_model = torch.jit.freeze(traced_model.eval())
                else:
                    traced_model = torch.jit.load(args.quantized_model_path)
                    traced_model = torch.jit.freeze(traced_model.eval())

                traced_model(**sample_inputs)
                traced_model(**sample_inputs)
            ipex._set_optimized_model_for_generation(
                self.model, optimized_model=traced_model
            )
        self.dataset = load_dataset("librispeech_asr", split="test.clean")
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
        )

    def _levenshtein(self, a: List, b: List) -> int:
        """Calculates the Levenshtein distance between a and b."""
        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n,m)) space
            a, b = b, a
            n, m = m, n

        current = list(range(n + 1))
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]

    def word_error_rate(self, hypotheses: List[str], references: List[str]) -> float:
        """
        Computes Average Word Error rate between two texts represented as
        corresponding lists of string. Hypotheses and references must have same length.

        Args:
            hypotheses: list of hypotheses
            references: list of references

        Returns:
            (float) average word error rate
        """
        scores = 0
        words = 0
        if len(hypotheses) != len(references):
            raise ValueError(
                "In word error rate calculation, hypotheses and reference"
                " lists must have the same number of elements. But I got:"
                "{0} and {1} correspondingly".format(len(hypotheses), len(references))
            )
        for h, r in zip(hypotheses, references):
            h_list = h.split()
            r_list = r.split()
            words += len(r_list)
            scores += self._levenshtein(h_list, r_list)
        if words != 0:
            wer = 1.0 * scores / words
        else:
            wer = float("inf")
        return wer, scores, words

    def evaluate(self):
        results = []
        references = []
        for batch_ndx, sample in enumerate(self.dataloader):
            inputs = sample["audio"]["array"].squeeze(0)
            model_inputs = self.tokenizer(
                inputs, sampling_rate=16000, return_tensors="pt"
            ).input_features
            with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True if self.amp_dtype == torch.bfloat16 else False,
            ):
                output = self.model.generate(
                    model_inputs,
                    do_sample=False,
                    temperature=0.9,
                    num_beams=self.num_beams,
                )
                gen_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                if len(results) == 0:
                    results = gen_text
                    references = sample["text"]
                else:
                    results += gen_text
                    references += sample["text"]
        references = [r.capitalize() for r in references]
        wer, scores, words = self.word_error_rate(results, references)
        return wer, scores, words


lm_tasks = []
lmms_tasks = []
other_tasks = []
lm_all_tasks = lm_eval.tasks.ALL_TASKS
try:
    initialize_tasks()
except Exception as e:
    print(e)
for task in args.tasks:
    if task in lm_all_tasks:
        lm_tasks.append(task)
    elif task in ALL_TASKS:
        lmms_tasks.append(task)
    elif task in ["librispeech_asr"]:
        other_tasks.append(task)
    else:
        print(f"Task {task} in not supported by lm_eval and lmms_eval")
        exit(0)
torch._C._jit_set_texpr_fuser_enabled(False)

if len(lm_tasks) != 0:
    lm_task_dict = lm_eval.tasks.get_task_dict(lm_tasks)
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
            add_special_tokens=False,
        )

    results = evaluator.evaluate(
        hfmodel,
        lm_task_dict,
        #        bootstrap_iters=1000,
        #        limit=100
    )
    print(evaluator.make_table(results))
elif len(lmms_tasks) != 0:
    task_names = lmms_utils.pattern_match(lmms_tasks, ALL_TASKS)
    lm = LMMS(
        pretrained=args.model,
        device="cpu",
        batch_size=args.batch_size,
        with_ipex=args.ipex,
        with_jit=not args.disable_jit,
        dtype=args.dtype,
        tp_number=world_size,
        config=args.config_file,
        add_special_tokens=False,
    )

    task_dict = lmms_eval.tasks.get_task_dict(task_names, model_name="test")
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if type(task_obj) == tuple:
            group, task_obj = task_obj
            if task_obj is None:
                continue
        lm.task_dict[task_name] = task_obj.dataset

        config = task_obj._config

    results = lmms_evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        # limit=10,
        # bootstrap_iters=100,
        cli_args=args,
    )
    print(lmms_evaluator.make_table(results))
elif len(other_tasks) != 0:
    if "librispeech_asr" in other_tasks:
        evaluator = LibriSpeech(
            pretrained=args.model,
            device="cpu",
            batch_size=args.batch_size,
            with_ipex=args.ipex,
            with_jit=not args.disable_jit,
            dtype=args.dtype,
            config=args.config_file,
            add_special_tokens=True,
            with_greedy=False,
        )
    wer, scores, num_words = evaluator.evaluate()
    print("Evaluation WER: {0}".format(wer))
    print("Accuracy: {:.15f} ".format(1 - wer))
