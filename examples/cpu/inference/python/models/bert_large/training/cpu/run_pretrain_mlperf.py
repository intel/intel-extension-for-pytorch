#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

"""BERT Pretraining"""

import argparse
import h5py
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import logging
import math
import random
import time

from concurrent.futures import ProcessPoolExecutor

from schedulers import LinearWarmupPolyDecayScheduler

import utils_local


from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForPreTraining,
    SchedulerType,
)

try:
    import oneccl_bindings_for_pytorch
except ImportError as e:
    oneccl_bindings_for_pytorch = False

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def get_eval_batchsize_per_worker(args):
    if torch.distributed.is_initialized():
        chunk_size = args.num_eval_examples // args.world_size
        rank = args.local_rank
        remainder = args.num_eval_examples % args.world_size
        if rank < remainder:
            return chunk_size + 1
        else:
            return chunk_size


def create_pretraining_dataset(
    input_file, max_pred_length, shared_list, args, worker_init_fn
):
    train_data = pretraining_dataset(
        input_file=input_file, max_pred_length=max_pred_length
    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    return train_dataloader, input_file


def create_eval_dataset(args, worker_init_fn):
    eval_data = []
    for eval_file in sorted(os.listdir(args.eval_dir)):
        eval_file_path = os.path.join(args.eval_dir, eval_file)

        if os.path.isfile(eval_file_path) and "part" in eval_file_path:
            eval_data.extend(
                pretraining_dataset(
                    eval_file_path, max_pred_length=args.max_predictions_per_seq
                )
            )
            if len(eval_data) > args.num_eval_examples:
                eval_data = eval_data[: args.num_eval_examples]
                break
    if torch.distributed.is_initialized():
        chunk_size = args.num_eval_examples // args.world_size
        rank = args.local_rank
        remainder = args.num_eval_examples % args.world_size
        if rank < remainder:
            eval_data = eval_data[
                (chunk_size + 1) * rank : (chunk_size + 1) * (rank + 1)
            ]
        else:
            eval_data = eval_data[
                chunk_size * rank + remainder : chunk_size * (rank + 1) + remainder
            ]

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0
    )
    return eval_dataloader


class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            (
                torch.from_numpy(input[index].astype(np.int64))
                if indice < 5
                else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            )
            for indice, input in enumerate(self.inputs)
        ]
        masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long) - 100
        index = self.max_pred_length
        masked_token_count = torch.count_nonzero(masked_lm_positions)
        if masked_token_count != 0:
            index = masked_token_count
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task"
    )

    # Required parameters
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain .hdf5 files  for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--eval_dir",
        default=None,
        type=str,
        help="The eval data dir. Should contain .hdf5 files  for the task.",
    )
    parser.add_argument(
        "--eval_iter_start_samples",
        default=3000000,
        type=int,
        help="Sample to begin performing eval.",
    )
    parser.add_argument(
        "--eval_iter_samples",
        default=-1,
        type=int,
        help="If set to -1, disable eval, \
                        else evaluate every eval_iter_samples during training",
    )
    parser.add_argument(
        "--num_eval_examples",
        default=10000,
        type=int,
        help="number of eval examples to run eval on",
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        help="The initial checkpoint to start training from.",
    )
    parser.add_argument(
        "--init_tf_checkpoint",
        default=None,
        type=str,
        help="The initial TF checkpoint to start training from.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        default=76,
        type=int,
        help="The maximum total of masked tokens in input sequence",
    )
    parser.add_argument(
        "--train_batch_size",
        default=18,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=128,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--weight_decay_rate",
        default=0.01,
        type=float,
        help="weight decay rate for LAMB.",
    )
    parser.add_argument(
        "--opt_lamb_beta_1", default=0.9, type=float, help="LAMB beta1."
    )
    parser.add_argument(
        "--opt_lamb_beta_2", default=0.999, type=float, help="LAMB beta2."
    )
    parser.add_argument(
        "--max_steps",
        default=1536,
        type=float,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_samples_termination",
        default=14000000,
        type=float,
        help="Total number of training samples to run.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.01,
        type=float,
        help="Proportion of optimizer update steps to perform linear learning rate warmup for. "
        "Typically 1/8th of steps for Phase2",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=float,
        help="Number of optimizer update steps to perform linear learning rate warmup for. "
        "Typically 1/8th of steps for Phase2",
    )
    parser.add_argument(
        "--start_warmup_step", default=0, type=float, help="Starting step for warmup. "
    )
    parser.add_argument(
        "--log_freq",
        type=float,
        default=10000.0,
        help="frequency of logging loss. If not positive, no logging is provided for training loss",
    )
    parser.add_argument(
        "--checkpoint_activations",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        action="store_true",
        help="Whether to resume training from checkpoint. If set, precedes init_checkpoint/init_tf_checkpoint",
    )
    parser.add_argument(
        "--keep_n_most_recent_checkpoints",
        type=int,
        default=20,
        help="Number of checkpoints to keep (rolling basis).",
    )
    parser.add_argument(
        "--num_samples_per_checkpoint",
        type=int,
        default=500000,
        help="Number of update steps until a model checkpoint is saved to disk.",
    )
    parser.add_argument(
        "--min_samples_to_start_checkpoints",
        type=int,
        default=3000000,
        help="Number of update steps until model checkpoints start saving to disk.",
    )
    parser.add_argument(
        "--skip_checkpoint",
        default=False,
        action="store_true",
        help="Whether to save checkpoints",
    )
    parser.add_argument(
        "--phase2",
        default=False,
        action="store_true",
        help="Only required for checkpoint saving format",
    )
    parser.add_argument(
        "--do_train",
        default=False,
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--bert_config_path",
        type=str,
        default="/workspace/phase1",
        help="Path bert_config.json is located in",
    )
    parser.add_argument(
        "--target_mlm_accuracy",
        type=float,
        default=0.72,
        help="Stop training after reaching this Masked-LM accuracy",
    )
    parser.add_argument(
        "--train_mlm_accuracy_window_size",
        type=int,
        default=0,
        help="Average accuracy over this amount of batches before performing a stopping criterion test",
    )
    parser.add_argument(
        "--num_epochs_to_generate_seeds_for",
        type=int,
        default=2,
        help="Number of epochs to plan seeds for. Same set across all workers.",
    )
    parser.add_argument(
        "--use_gradient_as_bucket_view",
        default=False,
        action="store_true",
        help="Turn ON gradient_as_bucket_view optimization in native DDP.",
    )
    parser.add_argument(
        "--dense_seq_output",
        default=False,
        action="store_true",
        help="Whether to run with optimizations.",
    )
    parser.add_argument(
        "--bf16", default=False, action="store_true", help="Enale BFloat16 training"
    )
    parser.add_argument(
        "--fp16", default=False, action="store_true", help="Enale Float16 training"
    )
    parser.add_argument(
        "--bf32", default=False, action="store_true", help="Enale BFloat32 training"
    )
    parser.add_argument(
        "--fp8", default=False, action="store_true", help="Enale FP8 training"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Whether to enable benchmark"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="Total batch size for training."
    )

    parser.add_argument(
        "--profile", action="store_true", help="Whether to enable profiling"
    )
    parser.add_argument("--ipex", action="store_true", default=False)
    parser.add_argument("--inductor", action="store_true", default=False)

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    # assert args.init_checkpoint is not None or args.init_tf_checkpoint is not None or found_resume_checkpoint(args), \
    #    "Must specify --init_checkpoint, --init_tf_checkpoint or have ckpt to resume from in --output_dir of the form *.pt"

    # assert not (args.init_checkpoint is not None and args.init_tf_checkpoint is not None), \
    #        "Can only specify one of --init_checkpoint and --init_tf_checkpoint"
    return args


def found_resume_checkpoint(args):
    if args.phase2:
        checkpoint_str = "pytorch_model.bin"
    else:
        checkpoint_str = "pytorch_model.bin"
    return (
        args.resume_from_checkpoint
        and len(glob.glob(os.path.join(args.output_dir, checkpoint_str))) > 0
    )


def setup_training(args):
    device = torch.device("cpu")
    if oneccl_bindings_for_pytorch and int(os.environ.get("PMI_SIZE", "0")) > 1:
        os.environ["RANK"] = os.environ.get("PMI_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", "1")
        torch.distributed.init_process_group(backend="ccl")
        device = torch.device("cpu")
        args.local_rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        print("##################Using CCL dist run", flush=True)
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
                args.gradient_accumulation_steps, args.train_batch_size
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not (args.do_train or (args.eval_dir and args.eval_iter_samples <= 0)):
        raise ValueError(" `do_train`  or should be in offline eval mode")

    if not args.resume_from_checkpoint or not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    return device, args


def prepare_model_and_optimizer(args, device):
    global_step = 0
    args.resume_step = 0
    checkpoint = None
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    config.dense_seq_output = args.dense_seq_output
    if args.model_name_or_path:
        model = AutoModelForPreTraining.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForPreTraining.from_config(config)
    # Load from Pyt checkpoint - either given as init_checkpoint, or picked up from output_dir if found
    # if args.init_checkpoint is not None or found_resume_checkpoint(args):
    #    # Prepare model
    #    #model = BertForPreTraining(config)
    #    model = BertForPreTrainingSegmented(config)

    #    # for k,v in model.state_dict().items():
    #    #     print(f'model-k,len(v)={k}, {v.numel()}')

    #    #model = BertForPretraining(config)
    #    if args.init_checkpoint is None: # finding checkpoint in output_dir
    #        assert False, "code path not tested with cuda graphs"
    #        checkpoint_str = "phase2_ckpt_*.pt" if args.phase2 else "phase1_ckpt_*.pt"
    #        model_names = [f for f in glob.glob(os.path.join(args.output_dir, checkpoint_str))]
    #        global_step = max([int(x.split('.pt')[0].split('_')[-1].strip()) for x in model_names])
    #        args.resume_step = global_step #used for throughput computation

    #        resume_init_checkpoint = os.path.join(args.output_dir, checkpoint_str.replace("*", str(global_step)))
    #        print("Setting init checkpoint to %s - which is the latest in %s" %(resume_init_checkpoint, args.output_dir))
    #        checkpoint=torch.load(resume_init_checkpoint, map_location="cpu")
    #    else:
    #        checkpoint=torch.load(args.init_checkpoint, map_location="cpu")["model"]
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "gamma", "beta", "LayerNorm"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay_rate,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.ipex:
        from intel_extension_for_pytorch.optim._lamb import Lamb

        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2),
            fused=True,
        )
    else:
        from lamb import Lamb

        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2),
        )

    if args.warmup_steps == 0:
        warmup_steps = int(args.max_steps * args.warmup_proportion)
        warmup_start = 0
    else:
        warmup_steps = args.warmup_steps
        warmup_start = args.start_warmup_step

    lr_scheduler = LinearWarmupPolyDecayScheduler(
        optimizer,
        start_warmup_steps=warmup_start,
        warmup_steps=warmup_steps,
        total_steps=args.max_steps,
        end_learning_rate=0.0,
        degree=1.0,
    )
    return model, optimizer, lr_scheduler, checkpoint, global_step


def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):
    global skipped_steps  # F824
    optimizer.step()
    global_step += 1
    return global_step


def run_eval(
    model,
    eval_dataloader,
    device,
    num_eval_examples,
    args,
    first_eval=False,
    use_cache=False,
):
    model.eval()
    total_eval_loss, total_eval_mlm_acc = 0.0, 0.0
    total_masked = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            (
                input_ids,
                segment_ids,
                input_mask,
                masked_lm_labels,
                next_sentence_labels,
            ) = batch
            outputs = None
            if args.bf16:
                with torch.autocast(
                    "cpu",
                ):
                    outputs = model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=masked_lm_labels,
                        next_sentence_label=next_sentence_labels,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=masked_lm_labels,
                    next_sentence_label=next_sentence_labels,
                )
            mlm_acc, num_masked = calc_mlm_acc(
                outputs, masked_lm_labels, args.dense_seq_output
            )
            total_eval_loss += outputs.loss.item() * num_masked
            total_eval_mlm_acc += mlm_acc * num_masked
            total_masked += num_masked
    model.train()
    total_masked = torch.tensor(total_masked, device=device, dtype=torch.int64)
    total_eval_loss = torch.tensor(total_eval_loss, device=device, dtype=torch.float64)
    if torch.distributed.is_initialized():
        # Collect total scores from all ranks
        torch.distributed.all_reduce(
            total_eval_mlm_acc, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(total_eval_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_masked, op=torch.distributed.ReduceOp.SUM)

    # Average by number of examples
    total_eval_mlm_acc /= total_masked
    total_eval_loss /= total_masked

    return total_eval_loss, total_eval_mlm_acc


def global_batch_size(args):
    return args.train_batch_size * args.gradient_accumulation_steps * args.world_size


def calc_mlm_acc(outputs, masked_lm_labels, dense_seq_output=False):
    prediction_scores = outputs.prediction_logits
    masked_lm_labels_flat = masked_lm_labels.view(-1)
    mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -100]
    if not dense_seq_output:
        prediction_scores_flat = prediction_scores.view(-1, prediction_scores.shape[-1])
        mlm_predictions_scores = prediction_scores_flat[masked_lm_labels_flat != -100]
        mlm_predictions = mlm_predictions_scores.argmax(dim=-1)
    else:
        mlm_predictions = prediction_scores.argmax(dim=-1)

    num_masked = mlm_labels.numel()
    mlm_acc = (mlm_predictions == mlm_labels).sum(dtype=torch.float) / num_masked

    return mlm_acc, num_masked


def calc_accuracy(outputs, masked_lm_labels, next_sentence_label, args):
    loss = outputs.loss.item()
    prediction_logits = outputs.prediction_logits
    seq_relationship_logits = outputs.seq_relationship_logits
    mlm_acc, num_masked = calc_mlm_acc(outputs, masked_lm_labels, args.dense_seq_output)
    seq_acc_t = (
        torch.argmax(seq_relationship_logits, dim=-1)
        .eq(next_sentence_label.view([-1]))
        .to(torch.float)
    )
    seq_acc_true, seq_tot = seq_acc_t.sum().item(), seq_acc_t.numel()
    seq_acc = seq_acc_true / seq_tot
    return loss, mlm_acc, num_masked, seq_acc, seq_tot


def main():
    args = parse_args()
    if not args.ipex and not args.inductor:
        print(
            "[Info] please specify --ipex or --inductor to choose path to run, exiting..."
        )
        exit(0)
    if args.ipex:
        print("Using ipex")
        import intel_extension_for_pytorch as ipex
        from intel_extension_for_pytorch.quantization.fp8 import (
            fp8_autocast,
            DelayedScaling,
            Format,
            prepare_fp8,
        )
    status = "aborted"  # later set to 'success' if termination criteria met
    device, args = setup_training(args)
    total_batch_size = global_batch_size(args)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args.local_rank == 0 or args.local_rank == -1:
        print("parsed args:")
        print(args)
    # Prepare optimizer
    (
        model,
        optimizer,
        lr_scheduler,
        checkpoint,
        global_step,
    ) = prepare_model_and_optimizer(args, device)
    model.train()
    if args.bf32 and args.ipex:
        ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
        model, optimizer = ipex.optimize(
            model, dtype=torch.float32, optimizer=optimizer, auto_kernel_selection=True
        )
    elif args.fp16 and args.ipex:
        scaler = torch.cpu.amp.GradScaler()
        model, optimizer = ipex.optimize(
            model,
            optimizer=optimizer,
            dtype=torch.half,
            auto_kernel_selection=True,
            weights_prepack=True,
            fuse_update_step=False,
        )
    elif args.bf16 and args.ipex:
        model, optimizer = ipex.optimize(
            model,
            optimizer=optimizer,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )
    elif args.fp8 and args.ipex:
        model, optimizer = prepare_fp8(model, optimizer)

    worker_seeds, shuffling_seeds = utils_local.setup_seeds(
        args.seed, args.num_epochs_to_generate_seeds_for, device
    )
    worker_seed = worker_seeds[args.local_rank]

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitObj(worker_seed)
    samples_trained = (
        global_step
        * args.train_batch_size
        * args.gradient_accumulation_steps
        * args.world_size
    )
    final_loss = float("inf")
    train_time_raw = float("inf")
    raw_train_start = time.time()
    if args.do_train:
        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 1
        training_steps = 0
        end_training, converged = False, False
        samples_trained_prev = 0

        # pre-compute eval boundaries
        samples_trained_per_step = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )
        start, stop, step = (
            args.eval_iter_start_samples,
            args.max_samples_termination,
            args.eval_iter_samples,
        )
        eval_steps = [
            math.ceil(i / samples_trained_per_step)
            for i in np.arange(start, stop, step)
        ]
        eval_count = 0
        next_eval_step = eval_steps[eval_count]
        pool = ProcessPoolExecutor(1)

        if args.target_mlm_accuracy:
            if args.train_mlm_accuracy_window_size > 0:
                accuracy_scores = []
                avg_mlm_accuracy = torch.Tensor([0])

        first_epoch = True
        if found_resume_checkpoint(args):
            f_start_id = checkpoint["files"][0]
            files = checkpoint["files"][1:]
            num_files = len(files)
        else:
            files = [
                os.path.join(args.input_dir, f)
                for f in os.listdir(args.input_dir)
                if os.path.isfile(os.path.join(args.input_dir, f)) and "part" in f
            ]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch % len(shuffling_seeds)]).shuffle(files)
            f_start_id = 0
    global skipped_steps  # F824
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            bucket_cap_mb=8192,
            gradient_as_bucket_view=args.use_gradient_as_bucket_view,
        )

    if args.inductor:
        if args.fp8 or args.bf32:
            print(
                "[Info] torch.compile() training does not support fp8 or bf32 yet, exiting..."
            )
            exit(0)
        from torch._inductor import config as inductor_config

        inductor_config.cpp_wrapper = True
        # torch._inductor.config.profiler_mark_wrapper_call = True
        # torch._inductor.config.cpp.enable_kernel_profile = True
        amp_dtype = torch.half if args.fp16 else torch.bfloat16
        with torch.autocast("cpu", enabled=args.bf16 or args.fp16, dtype=amp_dtype):
            print("[Info] Running training steps torch.compile() with default backend")
            model = torch.compile(model)

    now_step, now_skipped, skip_interval = 0, 0, 0
    # Start prefetching eval dataset
    if args.eval_dir:
        eval_dataset_future = pool.submit(
            create_eval_dataset, args, worker_init_fn=worker_init
        )
    # comparing to number of samples in a shard. There are ~38k samples in 4096-way shard, comparing to 10k to be safe
    need_next_training_shard = (
        args.train_batch_size * args.gradient_accumulation_steps * args.max_steps
        > 10000
    )
    print("Start Training.")
    while args.do_train and global_step < args.max_steps and not end_training:
        if args.local_rank == 0 or args.local_rank == -1:
            now_time = time.time()
            print("epoch:", epoch)

        thread = None

        # Reshuffle file list on subsequent epochs
        if not first_epoch:
            files = [
                os.path.join(args.input_dir, f)
                for f in os.listdir(args.input_dir)
                if os.path.isfile(os.path.join(args.input_dir, f)) and "part" in f
            ]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch % len(shuffling_seeds)]).shuffle(files)
            f_start_id = 0

        first_epoch = False

        shared_file_list = {}

        if args.world_size > num_files:
            remainder = args.world_size % num_files

        if torch.distributed.is_initialized() and args.world_size > num_files:
            data_file = files[
                (
                    f_start_id * args.world_size
                    + args.local_rank
                    + remainder * f_start_id
                )
                % num_files
            ]
        else:
            data_file = files[
                (f_start_id * args.world_size + args.local_rank) % num_files
            ]

        previous_file = data_file

        train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )
        send_lr_in_parallel = False
        lr_cpu = torch.tensor([0.0], dtype=torch.float32, device="cpu")
        completed_steps = 0
        bench_total_time = 0
        for f_id in range(f_start_id, len(files)):
            if args.world_size > num_files:
                data_file = files[
                    (f_id * args.world_size + args.local_rank + remainder * f_id)
                    % num_files
                ]
            else:
                data_file = files[
                    (f_id * args.world_size + args.local_rank) % num_files
                ]

            previous_file = data_file
            if need_next_training_shard:
                dataset_future = pool.submit(
                    create_pretraining_dataset,
                    data_file,
                    args.max_predictions_per_seq,
                    shared_file_list,
                    args,
                    worker_init_fn=worker_init,
                )
            t0 = time.time()
            for step, batch in enumerate(train_dataloader):
                training_steps += 1
                t_beg = time.time()
                t1 = time.time()
                (
                    input_ids,
                    segment_ids,
                    input_mask,
                    masked_lm_labels,
                    next_sentence_labels,
                ) = batch
                # print(f"Input shape: {batch['input_ids'].shape}")
                t2 = time.time()
                outputs = None
                if args.fp16:
                    with torch.autocast("cpu", enabled=True, dtype=torch.half):
                        outputs = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            labels=masked_lm_labels,
                            next_sentence_label=next_sentence_labels,
                        )
                elif args.bf16:
                    with torch.autocast(
                        "cpu",
                    ):
                        outputs = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            labels=masked_lm_labels,
                            next_sentence_label=next_sentence_labels,
                        )
                elif args.fp8 and args.ipex:
                    with fp8_autocast(
                        enabled=True,
                        calibrating=False,
                        fp8_recipe=DelayedScaling(fp8_format=Format.E4M3),
                    ):
                        outputs = model(
                            input_ids=input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            labels=masked_lm_labels,
                            next_sentence_label=next_sentence_labels,
                        )
                else:  # bf32 or fp32
                    outputs = model(
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=masked_lm_labels,
                        next_sentence_label=next_sentence_labels,
                    )
                t3 = time.time()
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                if args.fp16 and args.ipex:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                t4 = time.time()
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    if args.fp16 and args.ipex:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # progress_bar.update(1)
                t5 = time.time()
                t_end = time.time()
                completed_steps += 1
                if args.benchmark and completed_steps > 10:
                    bench_total_time = bench_total_time + (t_end - t_beg)
                if args.benchmark and completed_steps > 50:
                    throughput = 40 * args.train_batch_size / bench_total_time
                    print(
                        "Throughput: {:.3f} sentence/s".format(throughput), flush=True
                    )
                    if args.profile:
                        print("Running profiling ...")
                        with torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU],
                            record_shapes=True,
                        ) as p:
                            if args.fp16:
                                with torch.autocast(
                                    "cpu", enabled=True, dtype=torch.half
                                ):
                                    outputs = model(
                                        input_ids=input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=input_mask,
                                        labels=masked_lm_labels,
                                        next_sentence_label=next_sentence_labels,
                                    )
                            elif args.bf16:
                                with torch.autocast(
                                    "cpu",
                                ):
                                    outputs = model(
                                        input_ids=input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=input_mask,
                                        labels=masked_lm_labels,
                                        next_sentence_label=next_sentence_labels,
                                    )
                            elif args.fp8 and args.ipex:
                                with fp8_autocast(
                                    enabled=True,
                                    calibrating=False,
                                    fp8_recipe=DelayedScaling(fp8_format=Format.E4M3),
                                ):
                                    outputs = model(
                                        input_ids=input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=input_mask,
                                        labels=masked_lm_labels,
                                        next_sentence_label=next_sentence_labels,
                                    )
                            else:  # bf32 or fp32
                                outputs = model(
                                    input_ids=input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask,
                                    labels=masked_lm_labels,
                                    next_sentence_label=next_sentence_labels,
                                )
                            loss = outputs.loss
                            loss = loss / args.gradient_accumulation_steps
                            if args.fp16 and args.ipex:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                            if (
                                step % args.gradient_accumulation_steps == 0
                                or step == len(train_dataloader) - 1
                            ):
                                if args.fp16 and args.ipex:
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    optimizer.step()
                                lr_scheduler.step()
                                optimizer.zero_grad()

                        output = p.key_averages().table(sort_by="self_cpu_time_total")
                        print(output)
                    exit()

                gloss, lm_acc, num_masked, seq_acc, seq_tot = calc_accuracy(
                    outputs, masked_lm_labels, next_sentence_labels, args
                )

                print(
                    f"Step {training_steps:5d}: loss: {gloss:6.3f} lm_acc: {lm_acc:.3f} \
                    seq_acc: {seq_acc:.3f} lbs: {args.train_batch_size} gbs: {total_batch_size} \
                    DT: {(t1-t0)*1000.0:.1f} XT: {(t2-t1)*1000.0:.1f} FT: {(t3-t2)*1000.0:.1f} \
                    BT: {(t4-t3)*1000.0:.1f} OT: {(t5-t4)*1000.0:.1f} TT: {(t5-t0)*1000.0:.1f}"
                )

                update_step = training_steps % args.gradient_accumulation_steps == 0
                divisor = args.gradient_accumulation_steps
                if args.log_freq > 0:
                    average_loss += loss.item()
                if update_step:
                    now_lr = optimizer.param_groups[0]["lr"]
                    global_step += 1
                    if (
                        args.eval_dir
                        and args.eval_iter_samples > 0
                        and global_step == next_eval_step
                    ):
                        # on first eval, get eval_dataloader
                        if eval_count == 0:
                            eval_dataloader = create_eval_dataset(
                                args, worker_init_fn=worker_init
                            )  # eval_dataset_future.result(timeout=None)
                        samples_trained = (
                            global_step
                            * args.train_batch_size
                            * args.gradient_accumulation_steps
                            * args.world_size
                        )
                        samples_trained_prev = samples_trained
                        eval_avg_loss, eval_avg_mlm_accuracy = run_eval(
                            model,
                            eval_dataloader,
                            device,
                            args.num_eval_examples,
                            args,
                            first_eval=(eval_count == 0),
                        )
                        if args.local_rank == 0 or args.local_rank == -1:
                            print(
                                {
                                    "global_steps": global_step,
                                    "eval_loss": eval_avg_loss,
                                    "eval_mlm_accuracy": eval_avg_mlm_accuracy,
                                }
                            )

                            if args.target_mlm_accuracy:
                                if eval_avg_mlm_accuracy >= args.target_mlm_accuracy:
                                    end_training, converged = True, True
                                    if utils_local.is_main_process():
                                        print(
                                            "%f > %f, Target MLM Accuracy reached at %d"
                                            % (
                                                eval_avg_mlm_accuracy,
                                                args.target_mlm_accuracy,
                                                global_step,
                                            )
                                        )

                        eval_count += 1
                        next_eval_step = eval_steps[eval_count]
                if args.target_mlm_accuracy and args.train_mlm_accuracy_window_size > 0:
                    accuracy_scores.append(mlm_acc)
                    if update_step:
                        accuracy_scores = accuracy_scores[
                            -args.train_mlm_accuracy_window_size
                            * args.gradient_accumulation_steps :
                        ]
                        avg_mlm_accuracy[0] = sum(accuracy_scores) / len(
                            accuracy_scores
                        )
                        torch.distributed.all_reduce(
                            avg_mlm_accuracy, op=torch.distributed.ReduceOp.SUM
                        )
                        avg_mlm_accuracy /= args.world_size

                if (
                    args.log_freq > 0
                    and training_steps
                    % (args.log_freq * args.gradient_accumulation_steps)
                    == 0
                ):
                    samples_trained = (
                        global_step
                        * args.train_batch_size
                        * args.gradient_accumulation_steps
                        * args.world_size
                    )
                    if args.local_rank == 0 or args.local_rank == -1:
                        time_interval = time.time() - now_time
                        step_interval = global_step - now_step
                        now_time = time.time()
                        now_step = global_step
                        training_perf = (
                            args.train_batch_size
                            * args.gradient_accumulation_steps
                            * args.world_size
                            * (step_interval + skip_interval)
                            / time_interval
                        )
                        skip_interval = 0

                        if args.train_mlm_accuracy_window_size > 0:
                            print(
                                {
                                    "training_steps": training_steps,
                                    "average_loss": average_loss
                                    / (args.log_freq * divisor),
                                    "step_loss": loss.item()
                                    * args.gradient_accumulation_steps
                                    / divisor,
                                    "learning_rate": now_lr,
                                    "seq/s": training_perf,
                                    "global_steps": now_step,
                                    "samples_trained": samples_trained,
                                    "skipped_steps": now_skipped,
                                    "timestamp": now_time,
                                    "mlm_accuracy": avg_mlm_accuracy[0].item(),
                                }
                            )
                        else:
                            print(
                                {
                                    "training_steps": training_steps,
                                    "average_loss": average_loss
                                    / (args.log_freq * divisor),
                                    "step_loss": loss.item()
                                    * args.gradient_accumulation_steps
                                    / divisor,
                                    "learning_rate": now_lr,
                                    "seq/s": training_perf,
                                    "global_steps": now_step,
                                    "samples_trained": samples_trained,
                                    "skipped_steps": now_skipped,
                                    "timestamp": now_time,
                                }
                            )

                    average_loss = 0

                if global_step >= args.max_steps or end_training:
                    status = "success" if converged else "aborted"
                    end_training = True
                    train_time_raw = time.time() - raw_train_start
                    average_loss = torch.tensor(average_loss, dtype=torch.float32)
                    if args.log_freq > 0:
                        last_num_steps = (
                            int(training_steps / args.gradient_accumulation_steps)
                            % args.log_freq
                        )
                        last_num_steps = (
                            args.log_freq if last_num_steps == 0 else last_num_steps
                        )
                        average_loss = average_loss / (last_num_steps * divisor)
                    if torch.distributed.is_initialized():
                        average_loss /= args.world_size
                        torch.distributed.all_reduce(average_loss)
                    final_loss = average_loss.item()
                    if utils_local.is_main_process():
                        if args.train_mlm_accuracy_window_size > 0:
                            print(
                                (
                                    epoch,
                                    training_steps / args.gradient_accumulation_steps,
                                ),
                                {
                                    "final_loss": final_loss,
                                    "final_mlm_accuracy": avg_mlm_accuracy[0].item(),
                                },
                            )
                        else:
                            print(
                                (
                                    epoch,
                                    training_steps / args.gradient_accumulation_steps,
                                ),
                                {"final_loss": final_loss},
                            )

                if end_training or (
                    samples_trained - samples_trained_prev
                    >= args.num_samples_per_checkpoint
                    and samples_trained >= args.min_samples_to_start_checkpoints
                ):
                    samples_trained_prev = samples_trained
                    if utils_local.is_main_process() and not args.skip_checkpoint:
                        # Save a trained model
                        model.save_pretrained(args.output_dir)
                        model.config.to_json_file(args.output_dir + "config.json")

                        # model_to_save = model.module if hasattr(model,
                        #                                        'module') else model  # Only save the model it-self
                        # if args.phase2:
                        #    output_save_file = os.path.join(args.output_dir, "phase2_ckpt_{}.pt".format(samples_trained))
                        # else:
                        #    output_save_file = os.path.join(args.output_dir, "phase1_ckpt_{}.pt".format(samples_trained))
                        # if args.do_train:
                        #    torch.save({'model': model_to_save.state_dict(),
                        #                'optimizer': optimizer.state_dict(),
                        #                #'master params': list(amp.master_params(optimizer)),
                        #                'files': [f_id] + files}, output_save_file)
                        #
                        #    most_recent_ckpts_paths.append(output_save_file)
                        #    if len(most_recent_ckpts_paths) > args.keep_n_most_recent_checkpoints:
                        #        ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                        #        os.remove(ckpt_to_be_removed)

                    if samples_trained >= args.max_samples_termination or end_training:
                        status = "success" if converged else "aborted"
                        end_training = True
                        break
                t0 = time.time()

            del train_dataloader

            if samples_trained >= args.max_samples_termination or end_training:
                status = "success" if converged else "aborted"
                end_training = True
                break

            if not need_next_training_shard:
                dataset_future = pool.submit(
                    create_pretraining_dataset,
                    data_file,
                    args.max_predictions_per_seq,
                    shared_file_list,
                    args,
                    worker_init_fn=worker_init,
                )
            train_dataloader, data_file = dataset_future.result(timeout=None)
        epoch += 1

    return args, final_loss, train_time_raw


if __name__ == "__main__":
    main()
