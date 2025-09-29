#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ****************************************************************************

# ****************************************************************************
# BSD 3-Clause License
# # This model is originally from the PyTorch Examples repo
# (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
# ****************************************************************************

# ****************************************************************************
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ****************************************************************************

import argparse
import os
import random
import shutil
import time
import warnings
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
hub_model_names = torch.hub.list("facebookresearch/WSL-Images")
model_names += hub_model_names

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--hub", action="store_true", default=False, help="use model with torch hub"
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--warmup-epochs", type=float, default=5, help="number of warmup epochs"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument("--steps", default=-1, type=int, help="steps for validation")
parser.add_argument(
    "--training_steps", default=-1, type=int, help="steps for validation"
)
parser.add_argument("--base-op", type=str, default="sgd", help="base optimizer name")
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--base-lr", type=float, default=0.0125, help="learning rate for a single GPU"
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument("--epsilon", type=float, default=1e-5, help="epsilon for optimizer")
parser.add_argument(
    "--bn-bias-separately", action="store_true", default=True, help="skip bn and bias"
)
parser.add_argument(
    "--label-smoothing",
    type=float,
    default=0.1,
    help="label smoothing for cross entropy loss",
)
parser.add_argument(
    "--zero-init-residual",
    action="store_true",
    default=False,
    help="Initialize scale params in BN3 of a residual block to zeros instead ones. "
    "Improves accuracy by 0.2~0.3 percent according to https://arxiv.org/abs/1706.02677"
    "Used by Nvidia, but not part of MLPerf reference ",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--train-no-eval",
    action="store_true",
    help="only train, but not evaluate model on validation set",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--world-size", default=1, type=int, help="number of nodes for distributed training"
)
parser.add_argument(
    "--rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument("--port", default="29500", type=str, help="Port")
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument(
    "--ipex", action="store_true", default=False, help="use intel pytorch extension"
)
parser.add_argument(
    "--inductor",
    action="store_true",
    default=False,
    help="use torch.compile() default inductor backend",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disable CUDA"
)
parser.add_argument(
    "--int8", action="store_true", default=False, help="enable ipex int8 path"
)
parser.add_argument(
    "--bf16", action="store_true", default=False, help="enable ipex bf16 path"
)
parser.add_argument(
    "--bf32", action="store_true", default=False, help="enable ipex bf32 path"
)
parser.add_argument(
    "--fp16", action="store_true", default=False, help="enable ipex fp16 path"
)

parser.add_argument(
    "--jit", action="store_true", default=False, help="enable ipex jit fusionpath"
)
parser.add_argument(
    "--calibration",
    action="store_true",
    default=False,
    help="doing calibration step for int8 path",
)
parser.add_argument(
    "--configure-dir",
    default="configure.json",
    type=str,
    metavar="PATH",
    help="path to int8 configures, default file name is configure.json",
)
parser.add_argument(
    "--dummy",
    action="store_true",
    help="using  dummu data to test the performance of inference",
)
parser.add_argument(
    "-w",
    "--warmup-iterations",
    default=100,
    type=int,
    metavar="N",
    help="number of warmup iterati ons to run",
)
parser.add_argument(
    "--weight-sharing",
    action="store_true",
    default=False,
    help="using weight_sharing to test the performance of inference",
)
parser.add_argument(
    "--number-instance",
    default=0,
    type=int,
    help="the instance numbers for test the performance of latcy, only works when enable weight-sharing",
)

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    if args.inductor:
        from torch._inductor import config as inductor_config

        inductor_config.cpp_wrapper = True
        inductor_config.max_autotune = True
        inductor_config.max_autotune_gemm_backends = "CPP,ATEN"
        torch._dynamo.config.allow_unspec_int_on_nn_module = True

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None and args.cuda:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() if args.cuda else 0
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        os.environ["RANK"] = str(os.environ.get("PMI_RANK", args.rank))
        os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", args.world_size))
        os.environ["MASTER_PORT"] = args.port
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        # Initialize the process group with ccl backend
        if args.dist_backend == "ccl":
            dist.init_process_group(backend=args.dist_backend)
        else:
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
    if args.hub:
        torch.set_flush_denormal(True)
        model = torch.hub.load("facebookresearch/WSL-Images", args.arch)
    else:
        # create model
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    global num_classes
    num_classes = model.fc.out_features
    if args.ipex:
        import intel_extension_for_pytorch as ipex
    elif args.inductor:
        args.jit = False
    # for ipex path, always convert model to channels_last for bf16, fp32, int8.
    if args.ipex:
        model = model.to(memory_format=torch.channels_last)
    if args.inductor:
        model = model.to(memory_format=torch.channels_last)

    if args.ipex and args.bf32:
        ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
        print("using bf32 fmath mode\n")

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None and args.cuda:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            if args.cuda:
                model.cuda()
                print("create DistributedDataParallel in GPU")
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
            else:
                print("create DistributedDataParallel in CPU")
    elif args.gpu is not None and args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            if args.cuda:
                model.cuda()
        else:
            model = torch.nn.DataParallel(model)
            if args.cuda:
                model.cuda()

    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion = criterion.cuda(args.gpu)

    assert args.base_op.lower() == "sgd"
    print("Creating SGD optimizer")
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None and args.cuda:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None and args.cuda:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            if args.distributed and args.dist_backend == "ccl":
                model.load_state_dict(checkpoint["state_dict"])
            else:
                corrected_dict = {
                    k.replace("module.", "") if k.startswith("module.") else k: v
                    for k, v in checkpoint["state_dict"].items()
                }
                model.load_state_dict(corrected_dict)
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        cudnn.benchmark = True

    if args.weight_sharing:
        assert (
            args.dummy and args.batch_size
        ), "please using dummy data and set batch_size to 1 if you want run weight sharing case for latency case"
    if args.jit and args.int8:
        raise AssertionError("jit path is not available for int8 path using ipex")
    if args.calibration:
        assert (
            args.int8
        ), "please enable int8 path if you want to do int8 calibration path"
    if args.dummy:
        assert args.evaluate, "please using real dataset if you want run training path"
    if not args.ipex and not args.inductor:
        # for offical pytorch, int8 and jit path is not enabled.
        # for torch.compile(backend=inductor) INT8 quantization is been supported.
        assert not args.int8, "int8 path is not enabled for offical pytorch"
        assert not args.jit, "jit path is not enabled for offical pytorch"

    if not args.dummy:
        # Data loading code
        assert (
            args.data is not None
        ), "please set dataset path if you want to using real data"
        valdir = os.path.join(args.data, "val")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if not args.evaluate:
            traindir = os.path.join(args.data, "train")
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset
                )
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                num_workers=args.workers,
                pin_memory=True,
                sampler=train_sampler,
            )

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        train_loader = None
        val_loader = None

    if args.evaluate:
        if args.ipex:
            print("using ipex model to do inference\n")
        else:
            print("using offical pytorch model to do inference\n")

        # IPEX Path
        if args.ipex:
            model.eval()
            if args.int8:
                if not args.calibration:
                    from torch.ao.quantization import (
                        MinMaxObserver,
                        PerChannelMinMaxObserver,
                        QConfig,
                    )

                    x = torch.randn(args.batch_size, 3, 224, 224).contiguous(
                        memory_format=torch.channels_last
                    )
                    qconfig = QConfig(
                        activation=MinMaxObserver.with_args(
                            qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
                        ),
                        weight=PerChannelMinMaxObserver.with_args(
                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
                        ),
                    )
                    prepared_model = ipex.quantization.prepare(
                        model, qconfig, x, inplace=True
                    )
                    prepared_model.load_qconf_summary(qconf_summary=args.configure_dir)
                    model = ipex.quantization.convert(prepared_model)
                    model = torch.jit.trace(model, x)
                    model = torch.jit.freeze(model.eval())
                    y = model(x)
                    y = model(x)
                    print("running int8 evalation step\n")
            else:
                if args.bf16:
                    model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
                    print("running bfloat16 evalation step\n")
                elif args.fp16:
                    model = ipex.optimize(model, dtype=torch.half, inplace=True)
                    print("running float16 evalation step\n")
                else:
                    model = ipex.optimize(model, dtype=torch.float32, inplace=True)
                    print("running fp32 evalation step\n")
                if args.jit:
                    x = torch.randn(args.batch_size, 3, 224, 224).contiguous(
                        memory_format=torch.channels_last
                    )
                    if args.bf16:
                        x = x.to(torch.bfloat16)
                        with torch.autocast(
                            "cpu", dtype=torch.bfloat16
                        ), torch.no_grad():
                            model = torch.jit.trace(model, x).eval()
                    elif args.fp16:
                        x = x.to(torch.half)
                        with torch.autocast("cpu", dtype=torch.half), torch.no_grad():
                            model = torch.jit.trace(model, x).eval()
                    else:
                        with torch.no_grad():
                            model = torch.jit.trace(model, x).eval()
                    model = torch.jit.freeze(model)
        # torch.compile() inductor path
        elif args.inductor:
            model.eval()
            x = torch.randn(args.batch_size, 3, 224, 224).contiguous(
                memory_format=torch.channels_last
            )
            if args.int8:
                from torch.ao.quantization.quantize_pt2e import (
                    prepare_pt2e,
                    convert_pt2e,
                )
                import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
                from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
                    X86InductorQuantizer,
                )
                from torch.export import export_for_training

                print("[Info] Running torch.compile() INT8 quantization")
                with torch.no_grad():
                    example_inputs = (x,)
                    exported_model = export_for_training(model, example_inputs).module()
                    quantizer = X86InductorQuantizer()
                    quantizer.set_global(
                        xiq.get_default_x86_inductor_quantization_config()
                    )
                    prepared_model = prepare_pt2e(exported_model, quantizer)
                    if val_loader:
                        for i, (images, _) in enumerate(val_loader):
                            images = images.contiguous(
                                memory_format=torch.channels_last
                            )
                            prepared_model(images)
                            if i == 4:
                                break
                    else:
                        prepared_model(x)
                    converted_model = convert_pt2e(prepared_model)
                    torch.ao.quantization.move_exported_model_to_eval(converted_model)
                    if args.ipex:
                        print("[Info] Running torch.compile() with IPEX backend")
                        model = torch.compile(converted_model, backend="ipex")
                    else:
                        print("[Info] Running torch.compile() with default backend")
                        model = torch.compile(converted_model)
            elif args.bf16:
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                    x = x.to(torch.bfloat16)
                    if args.ipex:
                        print(
                            "[Info] Running torch.compile() BFloat16 with IPEX backend"
                        )
                        model = torch.compile(model, backend="ipex")
                    else:
                        print(
                            "[Info] Running torch.compile() BFloat16 with default backend"
                        )
                        model = torch.compile(model)
            elif args.fp16:
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.half):
                    x = x.to(torch.half)
                    if args.ipex:
                        print(
                            "[Info] Running torch.compile() FPloat16 with IPEX backend"
                        )
                        model = torch.compile(model, backend="ipex")
                    else:
                        print(
                            "[Info] Running torch.compile() FPloat16 with default backend"
                        )
                        model = torch.compile(model)
            else:
                with torch.no_grad():
                    if args.ipex:
                        print(
                            "[Info] Running torch.compile() Float32 with IPEX backend"
                        )
                        model = torch.compile(model, backend="ipex")
                    else:
                        print(
                            "[Info] Running torch.compile() Float32 with default backend"
                        )
                        model = torch.compile(model)
            with torch.no_grad(), torch.autocast(
                "cpu",
                enabled=args.bf16 or args.fp16,
                dtype=torch.half if args.fp16 else torch.bfloat16,
            ):
                y = model(x)
                y = model(x)
        validate(val_loader, model, criterion, args)
        return

    scaler = None
    if not args.distributed:
        # for bf32 path, calling ipex.optimize to calling ipex conv which enabled bf32 path
        if args.ipex and args.bf32:
            sample_input = torch.randn(args.batch_size, 3, 224, 224).contiguous(
                memory_format=torch.channels_last
            )
            model, optimizer = ipex.optimize(
                model,
                dtype=torch.float32,
                optimizer=optimizer,
                sample_input=sample_input,
            )

        if args.ipex and args.bf16:
            sample_input = torch.randn(args.batch_size, 3, 224, 224).contiguous(
                memory_format=torch.channels_last
            )
            model, optimizer = ipex.optimize(
                model,
                dtype=torch.bfloat16,
                optimizer=optimizer,
                weights_prepack=True,
                split_master_weight_for_bf16=False,
                sample_input=sample_input,
            )

        if args.ipex and args.fp16:
            scaler = torch.cpu.amp.GradScaler()
            model, optimizer = ipex.optimize(
                model, optimizer=optimizer, dtype=torch.half, fuse_update_step=False
            )

        if args.inductor:
            with torch.autocast(
                "cpu",
                enabled=args.bf16 or args.fp16,
                dtype=torch.half if args.fp16 else torch.bfloat16,
            ):
                if args.ipex:
                    print(
                        "[Info] Running training steps torch.compile() with IPEX backend"
                    )
                    model = torch.compile(model, backend="ipex")
                else:
                    print(
                        "[Info] Running training steps torch.compile() with default backend"
                    )
                    model = torch.compile(model)

    # parallelize
    if args.distributed and not args.cuda and args.gpu is None:
        print("create DistributedDataParallel in CPU")
        device_ids = None
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    num_steps_per_epoch = len(train_loader)
    if args.base_op.lower() == "sgd":
        lr_scheduler = None
    train(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        args,
        train_sampler,
        ngpus_per_node,
        scaler,
    )


def train(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    lr_scheduler,
    args,
    train_sampler,
    ngpus_per_node,
    scaler,
):
    global best_acc1
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader) * (args.epochs - args.start_epoch),
        [batch_time, data_time, losses, top1, top5],
        prefix="Training: ",
    )

    # switch to train mode
    model.train()

    if args.bf16:
        print("running bfloat16 training step\n")
    elif args.fp16:
        print("running float16 training step\n")
    else:
        print("running fp32 training step\n")
    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.base_op.lower() == "sgd":
            adjust_learning_rate(optimizer, epoch, args)

        for i, (images, target) in enumerate(train_loader):
            if (
                args.training_steps > 0
                and (epoch - args.start_epoch) * len(train_loader) + i
                >= args.training_steps
            ):
                break
            # measure data loading time
            if args.ipex:
                images = images.contiguous(memory_format=torch.channels_last)
            if args.bf16:
                images = images.to(torch.bfloat16)
            if args.fp16:
                images = images.to(torch.half)

            if (epoch - args.start_epoch) * len(
                train_loader
            ) + i == args.warmup_iterations:
                print("begin collecting time................................")
            if (epoch - args.start_epoch) * len(
                train_loader
            ) + i >= args.warmup_iterations:
                data_time.update(time.time() - end)
            # compute output
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            if args.bf16:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = model(images)
                output = output.to(torch.float32)
            elif args.fp16:
                with torch.autocast("cpu", dtype=torch.half):
                    output = model(images)
                output = output.to(torch.float32)

            else:
                output = model(images)
            loss = criterion(output, target)
            # compute gradient and do SGD step
            if not args.distributed:
                optimizer.zero_grad(set_to_none=True)
            if args.fp16:
                scaler.scale(loss).backward()
                # TODO check it works for fp16 dtype
                if lr_scheduler:
                    lr_scheduler.step()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if lr_scheduler:
                    lr_scheduler.step()
                optimizer.step()
            # measure elapsed time
            if (epoch - args.start_epoch) * len(
                train_loader
            ) + i >= args.warmup_iterations:
                batch_time.update(time.time() - end)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if (
                (epoch - args.start_epoch) * len(train_loader) + i
            ) % args.print_freq == 0:
                progress.display((epoch - args.start_epoch) * len(train_loader) + i)

            if (epoch - args.start_epoch) * len(
                train_loader
            ) + i >= args.warmup_iterations - 1:
                end = time.time()

        if not args.train_no_eval:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
            ):
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_acc1,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                )

    batch_size = args.batch_size
    perf = batch_size / (batch_time.avg - data_time.avg)
    print("Training throughput: {:.3f} fps".format(perf))


def run_weights_sharing_model(m, tid, args):
    steps = args.steps if args.steps > 0 else 300
    start_time = time.time()
    num_images = 0
    time_consume = 0
    timeBuff = []
    x = torch.randn(args.batch_size, 3, 224, 224)
    if args.bf16:
        x = x.to(torch.bfloat16)
    if args.fp16:
        x = x.to(torch.half)
    x = x.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        while num_images < steps:
            start_time = time.time()
            if not args.jit and args.bf16:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    y = m(x)
            elif not args.jit and args.fp16:
                with torch.autocast("cpu", dtype=torch.half):
                    y = m(x)
            else:
                y = m(x)

            end_time = time.time()
            if num_images > args.warmup_iterations:
                time_consume += end_time - start_time
                timeBuff.append(end_time - start_time)
            num_images += 1
        fps = (steps - args.warmup_iterations) / time_consume
        avg_time = time_consume * 1000 / (steps - args.warmup_iterations)
        timeBuff = np.asarray(timeBuff)
        p99 = np.percentile(timeBuff, 99)
        print("P99 Latency {:.2f} ms".format(p99 * 1000))
        print(
            "Instance num: %d Avg Time/Iteration: %f msec Throughput: %f fps"
            % (tid, avg_time, fps)
        )


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    if args.dummy:
        number_iter = args.steps if args.steps > 0 else 200
        if args.int8:
            number_iter = args.steps if args.steps > 0 else 200
    else:
        number_iter = args.steps if args.steps > 0 else len(val_loader)
    if args.calibration:
        number_iter = 100

    progress = ProgressMeter(
        number_iter, [batch_time, losses, top1, top5], prefix="Test: "
    )
    print("Evaluating RESNET: total Steps: {}".format(number_iter))

    # switch to evaluate mode
    if not (args.inductor and args.int8):
        model.eval()

    if args.ipex and args.int8 and args.calibration:
        print("runing int8 calibration step\n")
        import intel_extension_for_pytorch as ipex
        from torch.ao.quantization import (
            MinMaxObserver,
            PerChannelMinMaxObserver,
            QConfig,
        )

        qconfig = QConfig(
            activation=MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
            ),
            weight=PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric
            ),
        )
        x = torch.randn(1, 3, 224, 224)
        prepared_model = ipex.quantization.prepare(model, qconfig, x, inplace=True)
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.contiguous(memory_format=torch.channels_last)
                prepared_model(images)
                if i == 4:
                    print(i)
                    break
            prepared_model.save_qconf_summary(args.configure_dir)
            print(".........calibration step done..........")
    else:
        if args.dummy:
            # always running channle last for fp32, bf16, int8
            with torch.no_grad():
                if args.weight_sharing:
                    threads = []
                    for i in range(1, args.number_instance + 1):
                        thread = threading.Thread(
                            target=run_weights_sharing_model, args=(model, i, args)
                        )
                        threads.append(thread)
                        thread.start()
                    for thread in threads:
                        thread.join()
                    exit()
                else:
                    images = torch.randn(args.batch_size, 3, 224, 224).contiguous(
                        memory_format=torch.channels_last
                    )
                    target = (torch.rand(args.batch_size) * num_classes).long()
                    if args.bf16:
                        images = images.to(torch.bfloat16)
                    if args.fp16:
                        images = images.to(torch.half)
                    for i in range(number_iter):
                        if i >= args.warmup_iterations:
                            end = time.time()
                        if not args.jit and args.bf16:
                            with torch.autocast("cpu", dtype=torch.bfloat16):
                                output = model(images)
                        elif not args.jit and args.fp16:
                            with torch.autocast("cpu", dtype=torch.half):
                                output = model(images)
                        else:
                            output = model(images)

                        if i >= args.warmup_iterations:
                            batch_time.update(time.time() - end)

                        if args.bf16 or args.fp16:
                            output = output.to(torch.float32)

                        loss = criterion(output, target)
                        # measure accuracy and record loss
                        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        losses.update(loss.item(), images.size(0))
                        top1.update(acc1[0], images.size(0))
                        top5.update(acc5[0], images.size(0))

                        if i % args.print_freq == 0:
                            progress.display(i)
        else:
            with torch.no_grad():
                for i, (images, target) in enumerate(val_loader):
                    end = time.time()
                    if args.ipex:
                        images = images.contiguous(memory_format=torch.channels_last)
                    if args.bf16:
                        images = images.to(torch.bfloat16)
                    if args.fp16:
                        images = images.to(torch.half)
                    if args.gpu is not None:
                        images = images.cuda(args.gpu, non_blocking=True)
                        if torch.cuda.is_available():
                            target = target.cuda(args.gpu, non_blocking=True)

                    if not args.jit and args.bf16:
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            output = model(images)
                    elif not args.jit and args.fp16:
                        with torch.autocast("cpu", dtype=torch.half):
                            output = model(images)

                    else:
                        output = model(images)

                    # compute output
                    batch_time.update(time.time() - end)
                    # print(output)
                    if args.bf16 or args.fp16:
                        output = output.to(torch.float32)
                    loss = criterion(output, target)
                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    if i % args.print_freq == 0:
                        progress.display(i)

        if args.weight_sharing:
            latency = stats.latency_avg_ms
            perf = stats.iters_per_second
        else:
            batch_size = args.batch_size
            latency = batch_time.avg / batch_size * 1000
            perf = batch_size / batch_time.avg

        print("inference latency %.3f ms" % latency)
        print("Throughput: {:.3f} fps".format(perf))
        print("Accuracy: {top1.avg:.3f} ".format(top1=top1))

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
