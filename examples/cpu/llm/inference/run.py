#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pathlib import Path
import argparse
from typing import List, Optional
import subprocess
import re


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generation script")

    # general arguments.
    parser.add_argument(
        "-m",
        "--model-name-or-path",
        type=str,
        help="huggingface model id or local directory containing model files",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        type=str,
        help="local specific model configuration file",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="bfloat16, float32",
    )
    parser.add_argument("--ipex", action="store_true")
    parser.add_argument("--output-dir", nargs="?", default="./saved_results")

    # quantization related arguments.
    parser.add_argument(
        "--quant-with-amp",
        action="store_true",
        help="by default static quant is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
    )
    parser.add_argument(
        "--quantized-model-path", default="", help="path to the quantized model file"
    )
    parser.add_argument(
        "--qconfig-summary-file", default="", help="qconfig for static quantization"
    )
    parser.add_argument("--quant-model-name", default="best_model.pt")

    parser.add_argument(
        "--dataset",
        nargs="?",
        default="NeelNanda/pile-10k",
        help="Calibration dataset for static quantization and GPTQ",
    )
    parser.add_argument(
        "--ipex-smooth-quant",
        action="store_true",
        help="smoothquant forstatic quantization",
    )
    parser.add_argument(
        "--calib-len",
        default=512,
        type=int,
        help="calibration dataset max or padding max length for SmoothQuant autotuning",
    )
    parser.add_argument(
        "--calib-iters",
        default=512,
        type=int,
        help="calibration iters for SmoothQuant autotuning",
    )
    parser.add_argument(
        "--calib-shuffle",
        action="store_true",
        help="whether to shuffle on calibration dataset for SmoothQuant autotuning",
    )
    parser.add_argument(
        "--calib-padding",
        action="store_true",
        help="whether to pad on calibration dataset for SmoothQuant autotuning",
    )
    parser.add_argument(
        "--calib-pad-val",
        default=1,
        type=int,
        help="calibration dataset padding value for SmoothQuant autotuning",
    )
    parser.add_argument(
        "--fallback-add",
        action="store_true",
        help="whether to fallback add ops to fp32 for SmoothQuant autotuning",
    )
    parser.add_argument("--alpha", default=0.5, help="alpha value for smoothquant")
    parser.add_argument(
        "--folding",
        default=False,
        type=bool,
        help="whether to fold mul into the previous layer",
    )
    parser.add_argument(
        "--init-alpha",
        default=0.5,
        type=float,
        help="a value to get baseline quantization error for auto-tuning",
    )
    parser.add_argument(
        "--alpha-min",
        default=0.0,
        type=float,
        help="min value of auto-tuning alpha search space",
    )
    parser.add_argument(
        "--alpha-max",
        default=1.0,
        type=float,
        help="max value of auto-tuning alpha search space",
    )
    parser.add_argument(
        "--alpha-step",
        default=0.1,
        type=float,
        help="step_size of auto-tuning alpha search space",
    )
    parser.add_argument(
        "--shared-criterion",
        choices=["min", "mean", "max"],
        default="max",
        type=str,
        help="criterion for input LayerNorm op of a transformer block",
    )
    parser.add_argument(
        "--enable-blockwise-loss",
        default=False,
        type=bool,
        help="whether to enable block-wise auto-tuning",
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
        choices=["INT8", "INT4", "NF4"],
        default="INT8",
        type=str,
        help="weight data type for weight only quantization. Unrelated to activation"
        " data type or lowp-mode. If `--gptq` is given, weight"
        " data type is always INT4 and this argument is not needed.",
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
        default="PER_IC_BLOCK",
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
        "--low-precision-checkpoint",
        default="",
        type=str,
        help="Low precision checkpoint file generated by calibration, such as GPTQ. It contains"
        " modified weights, scales, zero points, etc. For better accuracy of weight only"
        " quantization with INT4 weight.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="Run GPTQ calibration to generate optimized INT4 weight for weight-only quantization."
        " This is recommended for INT4 to minimize accuracy drop after quantization.",
    )
    parser.add_argument(
        "--gptq-legacy-format",
        action="store_true",
        help="Indicate that the low-precision checkpoint is in the legacy format rather than the"
        " HuggingFace Optimum format for backward compatibility. It must be used with"
        " --low-precision-checkpoint. Otherwise, it has no effect.",
    )
    parser.add_argument(
        "--group-size",
        default=0,
        type=int,
        help="For GPTQ and weight-only quantization only. Group size defines granularity of quantization the"
        " along input channel of weight. The input channel size must be a multiple of the group size."
        " It is effective for both INT8 and INT4 weight dtype. It must be -1, 0 or a positive power of 2. -1 means"
        " group-size equals the input channel size (i.e., per-channel quantization). 0 means group-size is selected"
        " automatically, -1 for INT8 and 128 for INT4. If --low-precision-checkpoint is given, this parameter is "
        "overwritten by data in the checkpoint file.",
    )
    parser.add_argument(
        "--cache-weight-for-large-batch",
        action="store_true",
        help="Cache an extra linear weight for large batch inference, such as the first token (prefill phase)."
        " It brings better performance at the cost of higher memory usage. It is only valid for full bf16 path"
        " and weight-only quantization with lowp-mode=BF16. Otherwise, it has no effect.",
    )

    # inference related arguments.
    parser.add_argument(
        "--max-new-tokens", default=32, type=int, help="output max new tokens"
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--input-tokens", default="32", type=str)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="enable streaming mode for generation output (greedy search only)",
    )
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--num-iter", default=100, type=int, help="num iter")
    parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--token-latency", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--disable-deployment-mode", action="store_true")
    parser.add_argument(
        "--image-url", default=None, type=str, help="image url for image-to-text task"
    )
    parser.add_argument(
        "--audio",
        default=None,
        type=str,
        help="audio file for speech-to-text task",
    )
    # deepspeed inference related arguments.
    parser.add_argument("--autotp", action="store_true")
    parser.add_argument("--shard-model", action="store_true")
    parser.add_argument(
        "--local_rank", required=False, type=int, help="used by dist launchers"
    )
    parser.add_argument(
        "--lm-head-generation",
        action="store_true",
        help="Compute lm-head only for the last token in the sequence to speed up first token inference."
        " This argument is only needed for non-TP quantization cases. And note that in such cases,"
        " this feature is not compatible with lambada_openai accuracy test. If you want to run"
        " lambada_openai accuracy test with the quantized model afterwards, don't turn this feature on."
        " In other cases, this feature is always turned on regardless of this argument and it does not"
        " conflict with the accuracy test.",
    )
    args = parser.parse_args(args_in)

    parent_path = Path(__file__).parent.absolute()

    group_size = args.group_size
    if group_size == 0:
        # weight dtype is ignored if gptq is true
        if args.weight_dtype in ("INT4", "NF4"):
            group_size = 128
        else:
            group_size = -1
    assert group_size == -1 or (
        group_size > 0 and (group_size & (group_size - 1) == 0)
    ), f"Invalid group size for WOQ: {group_size}"

    if (
        re.search("llava", str(args.model_name_or_path), re.IGNORECASE)
        and args.prompt is None
    ):
        args.prompt = "What is this image?"
    if not args.autotp:
        if not args.ipex_weight_only_quantization and not args.ipex_smooth_quant:
            path = Path(parent_path, "single_instance/run_generation.py")
            infer_cmd = ["python", path]
            infer_cmd.extend(["-m", str(args.model_name_or_path)])
            infer_cmd.extend(["--dtype", str(args.dtype)])
            infer_cmd.extend(["--input-tokens", str(args.input_tokens)])
            infer_cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
            infer_cmd.extend(["--num-iter", str(args.num_iter)])
            infer_cmd.extend(["--num-warmup", str(args.num_warmup)])
            infer_cmd.extend(["--batch-size", str(args.batch_size)])

            if args.greedy:
                infer_cmd.extend(["--greedy"])
            if args.streaming:
                infer_cmd.extend(["--streaming"])
            if args.ipex:
                infer_cmd.extend(["--ipex"])
            if not args.disable_deployment_mode:
                infer_cmd.extend(["--deployment-mode"])
            if args.profile:
                infer_cmd.extend(["--profile"])
            if args.benchmark:
                infer_cmd.extend(["--benchmark"])
            if args.token_latency:
                infer_cmd.extend(["--token-latency"])

            if args.prompt is not None:
                infer_cmd.extend(["--prompt", str(args.prompt)])
            if args.config_file is not None:
                infer_cmd.extend(["--config-file", str(args.config_file)])
            if args.image_url is not None:
                infer_cmd.extend(["--image-url", str(args.image_url)])
            if args.cache_weight_for_large_batch:
                infer_cmd.extend(["--cache-weight-for-large-batch"])
            if args.audio is not None:
                infer_cmd.extend(["--audio", str(args.audio)])

            print("LLM RUNTIME INFO: running model geneartion...")
            result = subprocess.run(infer_cmd)
            if result.returncode != 0:
                print("LLM RUNTIME ERROR: Running generation task failed. Quit.")
                quit()
            print("LLM RUNTIME INFO: Finished successfully.")
        else:
            qpath = Path(parent_path, "single_instance/run_quantization.py")

            infer_cmd = ["python", qpath]
            # 1) quantization
            if args.quantized_model_path == "":
                quant_cmd = ["python", qpath]
                quant_cmd.extend(["-m", str(args.model_name_or_path)])
                quant_cmd.extend(["--output-dir", str(args.output_dir)])
                quant_cmd.extend(["--input-tokens", str(args.input_tokens)])
                quant_cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
                if args.config_file is not None:
                    quant_cmd.extend(["--config-file", str(args.config_file)])
                if args.quant_with_amp:
                    quant_cmd.extend(["--quant-with-amp"])
                if args.greedy:
                    quant_cmd.extend(["--greedy"])
                if args.image_url is not None:
                    quant_cmd.extend(["--image-url", str(args.image_url)])
                if args.cache_weight_for_large_batch:
                    quant_cmd.extend(["--cache-weight-for-large-batch"])
                if args.audio is not None:
                    quant_cmd.extend(["--audio", str(args.audio)])
                if args.lm_head_generation:
                    print(
                        "LLM RUNTIME WARNING: `--lm-head-generation` is set. You cannot use the "
                        "quantized model for lamababa_openai accuracy test"
                    )
                    quant_cmd.extend(["--lm-head-generation"])
                if args.ipex_weight_only_quantization:
                    quant_cmd.extend(["--ipex-weight-only-quantization"])
                    quant_cmd.extend(["--weight-dtype", str(args.weight_dtype)])
                    quant_cmd.extend(["--lowp-mode", str(args.lowp_mode)])
                    quant_cmd.extend(["--act-quant-mode", str(args.act_quant_mode)])
                    if args.gptq:
                        print(
                            "LLM RUNTIME INFO: Weight dtype set to INT4 since `--gptq` is sepcified"
                            " and `--weight-dtype` is ignored."
                        )
                        if args.low_precision_checkpoint == "":
                            gptq_cmd = [
                                "python",
                                Path(parent_path, "utils/run_gptq.py"),
                            ]
                            gptq_cmd.extend(["--model", str(args.model_name_or_path)])
                            gptq_cmd.extend(["--dataset", str(args.dataset)])
                            gptq_cmd.extend(["--group-size", str(group_size)])
                            gptq_cmd.extend(["--output-dir", str(args.output_dir)])
                            print(
                                "LLM RUNTIME INFO: Running GPTQ calibration with group_size {}...".format(
                                    group_size
                                )
                            )
                            result = subprocess.run(gptq_cmd)
                            if result.returncode != 0:
                                print(
                                    "LLM RUNTIME ERROR: Running GPTQ calibration failed. Quit."
                                )
                                quit()
                            print(
                                "LLM RUNTIME INFO: Running GPTQ calibration finished."
                            )
                            quant_cmd.extend(
                                [
                                    "--low-precision-checkpoint",
                                    str(args.output_dir)
                                    + f"/gptq_checkpoint_g{group_size}.pt",
                                ]
                            )
                        else:
                            quant_cmd.extend(
                                [
                                    "--low-precision-checkpoint",
                                    str(args.low_precision_checkpoint),
                                ]
                            )
                            if args.gptq_legacy_format:
                                quant_cmd.extend(["--gptq-legacy-format"])
                    elif args.low_precision_checkpoint != "":
                        quant_cmd.extend(
                            [
                                "--low-precision-checkpoint",
                                str(args.low_precision_checkpoint),
                            ]
                        )
                        if args.gptq_legacy_format:
                            quant_cmd.extend(["--gptq-legacy-format"])
                    else:
                        # No need to set group size if args.gptq is true
                        # Group size is read from the checkpoint
                        quant_cmd.extend(["--group-size", str(group_size)])
                else:
                    quant_cmd.extend(["--ipex-smooth-quant"])
                    quant_cmd.extend(["--calib-len", str(args.calib_len)])
                    quant_cmd.extend(["--calib-iters", str(args.calib_iters)])
                    if args.calib_shuffle:
                        quant_cmd.extend(["--calib-shuffle"])
                    if args.calib_padding:
                        quant_cmd.extend(["--calib-padding"])
                    quant_cmd.extend(["--calib-pad-val", str(args.calib_pad_val)])
                    if args.fallback_add:
                        quant_cmd.extend(["--fallback-add"])
                    quant_cmd.extend(["--alpha", str(args.alpha)])
                    if args.folding:
                        quant_cmd.extend(["--folding"])
                    quant_cmd.extend(["--init-alpha", str(args.init_alpha)])
                    quant_cmd.extend(["--alpha-min", str(args.alpha_min)])
                    quant_cmd.extend(["--alpha-max", str(args.alpha_max)])
                    quant_cmd.extend(["--alpha-step", str(args.alpha_step)])
                    quant_cmd.extend(["--shared-criterion", str(args.shared_criterion)])
                    if args.enable_blockwise_loss:
                        quant_cmd.extend(["--enable-blockwise-loss"])
                    quant_cmd.extend(["--dataset", str(args.dataset)])
                    quant_cmd.extend(
                        ["--qconfig-summary-file", str(args.qconfig_summary_file)]
                    )
                print("LLM RUNTIME INFO: quantizing model ...")
                result = subprocess.run(quant_cmd)
                if result.returncode != 0:
                    print("LLM RUNTIME ERROR: Quantizing model failed. Quit.")
                    quit()
                print(
                    "LLM RUNTIME INFO: Model quantized successfully, saved to {}.".format(
                        str(args.output_dir) + "/best_model.pt"
                    )
                )
                infer_cmd.extend(
                    [
                        "--quantized-model-path",
                        str(args.output_dir) + "/" + str(args.quant_model_name),
                    ]
                )
            else:
                infer_cmd.extend(
                    ["--quantized-model-path", str(args.quantized_model_path)]
                )

            # 2) inference
            infer_cmd.extend(["-m", str(args.model_name_or_path)])
            infer_cmd.extend(["--input-tokens", str(args.input_tokens)])
            infer_cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
            infer_cmd.extend(["--num-iter", str(args.num_iter)])
            infer_cmd.extend(["--num-warmup", str(args.num_warmup)])
            infer_cmd.extend(["--batch-size", str(args.batch_size)])

            if args.quant_with_amp:
                infer_cmd.extend(["--quant-with-amp"])
            if args.greedy:
                infer_cmd.extend(["--greedy"])
            if args.streaming:
                infer_cmd.extend(["--streaming"])
            if args.profile:
                infer_cmd.extend(["--profile"])
            if args.benchmark:
                infer_cmd.extend(["--benchmark"])
            if args.token_latency:
                infer_cmd.extend(["--token-latency"])
            if args.image_url is not None:
                infer_cmd.extend(["--image-url", str(args.image_url)])
            if args.audio is not None:
                infer_cmd.extend(["--audio", str(args.audio)])

            if args.prompt is not None:
                infer_cmd.extend(["--prompt", str(args.prompt)])
            if args.config_file is not None:
                infer_cmd.extend(["--config-file", str(args.config_file)])

            print("LLM RUNTIME INFO: running model geneartion...")
            result = subprocess.run(infer_cmd)
            if result.returncode != 0:
                print("LLM RUNTIME ERROR: Running generation task failed. Quit.")
                quit()
            print("LLM RUNTIME INFO: Finished successfully.")

    else:
        path = Path(parent_path, "distributed/run_generation_with_deepspeed.py")
        infer_cmd = ["python", path]
        if args.shard_model:
            spath = Path(parent_path, "utils/create_shard_model.py")
            shard_cmd = ["python", spath]
            shard_cmd.extend(["-m", str(args.model_name_or_path)])
            MODEL_CLASSES = {
                "gpt-j": ("/gptj_local_shard"),
                "gpt-neox": ("/gptneox_local_shard"),
                "llava": ("/llava_local_shard"),
                "llama": ("/llama_local_shard"),
                "opt": ("/opt_local_shard"),
                "falcon": ("/falcon_local_shard"),
                "bloom": ("/bloom_local_shard"),
                "codegen": ("/codegen_local_shard"),
                "baichuan": ("/baichuan_local_shard"),
                "chatglm": ("/chatglm_local_shard"),
                "starcoder": ("/starcoder_local_shard"),
                "t5": ("/t5_local_shard"),
                "mixtral": ("/mixtral_local_shard"),
                "mistral": ("/mistral_local_shard"),
                "mpt": ("/mpt_local_shard"),
                "stablelm": ("/stablelm_local_shard"),
                "dolly": ("/dolly_local_shard"),
                "qwen": ("/qwen_local_shard"),
                "git": ("/git_local_shard"),
                "yuan": ("/yuan_local_shard"),
                "phi-3": ("/phi-3_local_shard"),
                "phi": ("/phi_local_shard"),
                "whisper": ("/whisper_local_shard"),
            }
            model_type = next(
                (
                    x
                    for x in MODEL_CLASSES.keys()
                    if x in args.model_name_or_path.lower()
                ),
                "auto",
            )
            work_path = Path(str(args.output_dir))
            if not work_path.exists():
                Path.mkdir(work_path)
                model_path = Path(str(args.output_dir) + str(MODEL_CLASSES[model_type]))
                if not model_path.exists():
                    Path.mkdir(model_path)
            shard_cmd.extend(
                ["--save-path", str(args.output_dir) + str(MODEL_CLASSES[model_type])]
            )
            if args.local_rank is not None:
                shard_cmd.extend(["--local_rank", str(args.local_rank)])
            print("LLM RUNTIME INFO: sharding model...")
            result = subprocess.run(shard_cmd)
            if result.returncode != 0:
                print("LLM RUNTIME ERROR: Sharding model failed. Quit.")
                quit()
            print("LLM RUNTIME INFO: Model sharded successfully.")
            # use absolute path here to avoid path error in deepspeed
            infer_cmd.extend(
                [
                    "-m",
                    str(os.path.abspath(args.output_dir))
                    + str(MODEL_CLASSES[model_type]),
                ]
            )
        else:
            model_name_or_path = args.model_name_or_path
            if os.path.exists(model_name_or_path):
                # use absolute path here to avoid path error in deepspeed
                model_name_or_path = os.path.abspath(model_name_or_path)
            infer_cmd.extend(["-m", str(model_name_or_path)])

        infer_cmd.extend(["--dtype", str(args.dtype)])
        infer_cmd.extend(["--input-tokens", str(args.input_tokens)])
        infer_cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
        infer_cmd.extend(["--num-iter", str(args.num_iter)])
        infer_cmd.extend(["--num-warmup", str(args.num_warmup)])
        infer_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.local_rank is not None:
            infer_cmd.extend(["--local_rank", str(args.local_rank)])
        if args.greedy:
            infer_cmd.extend(["--greedy"])
        if args.streaming:
            infer_cmd.extend(["--streaming"])
        if args.ipex:
            infer_cmd.extend(["--ipex"])
        if not args.disable_deployment_mode:
            infer_cmd.extend(["--deployment-mode"])
        if args.profile:
            infer_cmd.extend(["--profile"])
        if args.benchmark:
            infer_cmd.extend(["--benchmark"])
        if args.token_latency:
            infer_cmd.extend(["--token-latency"])
        if args.image_url is not None:
            infer_cmd.extend(["--image-url", str(args.image_url)])
        if args.audio is not None:
            infer_cmd.extend(["--audio", str(args.audio)])

        if args.prompt is not None:
            infer_cmd.extend(["--prompt", str(args.prompt)])
        if args.config_file is not None:
            infer_cmd.extend(["--config-file", str(args.config_file)])

        if args.ipex_weight_only_quantization:
            infer_cmd.extend(["--ipex-weight-only-quantization"])
            infer_cmd.extend(["--weight-dtype", str(args.weight_dtype)])
            infer_cmd.extend(["--lowp-mode", str(args.lowp_mode)])
            infer_cmd.extend(["--group-size", str(group_size)])
            if args.quant_with_amp:
                infer_cmd.extend(["--quant-with-amp"])
        if args.cache_weight_for_large_batch:
            infer_cmd.extend(["--cache-weight-for-large-batch"])

        print("LLM RUNTIME INFO: running model geneartion with deepspeed (autotp)...")
        result = subprocess.run(infer_cmd)
        if result.returncode != 0:
            print("LLM RUNTIME ERROR: Running generation task failed. Quit.")
            quit()
        print("LLM RUNTIME INFO: Finished successfully.")


if __name__ == "__main__":
    main()
