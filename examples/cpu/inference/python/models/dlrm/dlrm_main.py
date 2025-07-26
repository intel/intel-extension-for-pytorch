#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
#

import argparse
import itertools
import os
import sys
from torch.profiler import record_function, ProfilerActivity
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import List, Optional
import time

import torch
import torch.distributed as dist

print(f"torch version {torch.__version__}")
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES

from torchrec.models.dlrm import DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from tqdm import tqdm

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm/data:dlrm_dataloader
    from data_process.dlrm_dataloader import get_dataloader

    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:multi_hot
    from multi_hot import Multihot, RestartableMap
except ImportError:
    pass

# internal import
try:
    from .data_process.dlrm_dataloader import get_dataloader  # noqa F811
    from .multi_hot import Multihot, RestartableMap  # noqa F811
except ImportError:
    pass

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.
# Optimizer parameters:
ADAGRAD_LR_DECAY = 0
ADAGRAD_INIT_ACC = 0
ADAGRAD_EPS = 1e-8
WEIGHT_DECAY = 0


def unpack(input) -> dict:
    output = {}
    for k, v in input.to_dict().items():
        output[k] = {}
        output[k]["values"] = v._values.int()
        output[k]["offsets"] = v._offsets.int()
    return output


class InteractionType(Enum):
    ORIGINAL = "original"
    DCN = "dcn"
    PROJECTION = "projection"

    def __str__(self):
        return self.value


def load_snapshot(model, snapshot_dir):
    from torchsnapshot import Snapshot

    snapshot = Snapshot(path=snapshot_dir)
    snapshot.restore(app_state={"model": model})


def trace_handler(prof):
    profile_dir = "./"
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    trace_dir = (
        f"{profile_dir}/{prof._mode}-{prof._dtype}-dlrm_trace_step{prof.step_num}.json"
    )
    print(trace_dir)
    prof.export_chrome_trace(trace_dir)


prof_schedule = torch.profiler.schedule(wait=10, warmup=10, active=1, repeat=10)


def print_memory(stage):
    import os
    import psutil

    print(
        f"dlrmv2-memory-usage-log: {time.time()}, {stage}, {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024}"
    )


def fetch_batch(dataloader):
    try:
        batch = dataloader.dataset.load_batch()
    except:  # noqa B001
        import torchrec

        dataset = dataloader.source.dataset
        if isinstance(
            dataset, torchrec.datasets.criteo.InMemoryBinaryCriteoIterDataPipe
        ):
            sample_list = list(range(dataset.batch_size))
            dense = dataset.dense_arrs[0][sample_list, :]
            sparse = [arr[sample_list, :] for arr in dataset.sparse_arrs][
                0
            ] % dataset.hashes
            labels = dataset.labels_arrs[0][sample_list, :]
            return dataloader.func(dataset._np_arrays_to_batch(dense, sparse, labels))
        batch = dataloader.func(
            dataloader.source.dataset.batch_generator._generate_batch()
        )
    return batch


def split_dense_input_and_label_for_ranks(batch):
    my_rank = dist.get_rank()
    my_size = dist.get_world_size()
    local_bs = int(batch.dense_features.shape[0] / my_size)
    start = local_bs * my_rank
    end = start + local_bs
    batch.dense_features = batch.dense_features[start:end]
    batch.labels = batch.labels[start:end]
    return batch


def parse_autocast(dtype: str):
    _dtype = None
    autocast = False
    if dtype == "bf16":
        autocast = True
        _dtype = torch.bfloat16
    elif dtype == "fp16":
        autocast = True
        _dtype = torch.float16
    elif dtype == "int8":
        _dtype = torch.int8
    else:
        assert dtype in ["fp32", "bf32", "tf32"]
        _dtype = torch.float
    return autocast, _dtype


def convert_int8(args, model, dataloader):
    from torch.ao.quantization import (
        HistogramObserver,
        PerChannelMinMaxObserver,
        QConfig,
    )
    from intel_extension_for_pytorch.quantization import prepare, convert

    qconfig = QConfig(
        activation=HistogramObserver.with_args(
            qscheme=torch.per_tensor_symmetric,
            dtype=torch.qint8,
            bins=127,
            quant_min=-127,
            quant_max=126,
        ),
        weight=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )
    batch = fetch_batch(dataloader)
    batch.sparse_features = unpack(batch.sparse_features)
    print_memory("int8 prepare")
    model = prepare(
        model,
        qconfig,
        example_inputs=(batch.dense_features, batch.sparse_features),
        inplace=True,
    )
    if args.calibration:
        # https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch#calibration-set
        assert args.synthetic_multi_hot_criteo_path, "need real dataset to calibrate"
        batch_idx = list(range(128000))
        batch = dataloader.dataset.load_batch(batch_idx)
        batch.sparse_features = unpack(batch.sparse_features)
        model(batch.dense_features, batch.sparse_features)
        model.save_qconf_summary(qconf_summary=args.int8_configure_dir)
        print(f"calibration done and save to {args.int8_configure_dir}")
        exit()
    else:
        model.load_qconf_summary(qconf_summary=args.int8_configure_dir)
        print_memory("int8 convert")
        convert(model, inplace=True)
        model.eval()
        torch._C._jit_set_texpr_fuser_enabled(False)
        print_memory("int8 trace")
        model = torch.jit.trace(
            model, (batch.dense_features, batch.sparse_features), check_trace=True
        )
        print_memory("int8 freeze")
        model = torch.jit.freeze(model)
        print_memory("int8 jit optimize")
        model(batch.dense_features, batch.sparse_features)
        model(batch.dense_features, batch.sparse_features)
        return model


def aoti_benchmark_compile(ninstances, nbatches, bs, tmp_dir, target_dir):
    import textwrap

    inference_template = textwrap.dedent(
        """
        #include <vector>

        #include <torch/torch.h>
        #include <torch/script.h>
        #include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

        #include <iostream>

        int main() {
            c10::InferenceMode mode;
            size_t ninstances = %s;
            if (ninstances == 0) ninstances = 1;
            size_t niters = %s;
            size_t total_iters = ninstances * niters;
            size_t bs = %s;
            auto module = torch::jit::load("%s");
            std::vector<torch::Tensor> _input_vec = module.attr("tensor_list").toTensorList().vec();
            std::vector<torch::Tensor> input_vec;
            for (const auto t : _input_vec) {
                input_vec.push_back(t.clone());
            }
            torch::inductor::AOTIModelContainerRunnerCpu runner("%s", ninstances * 2);
            std::vector<torch::Tensor> outputs = runner.run(input_vec);

            using Input = std::vector<torch::Tensor>;
            std::vector<std::vector<Input>> thread_inputs(ninstances);
            std::vector<size_t> input_iters(ninstances);
            for (const auto thread_id : c10::irange(ninstances)) {
                for (const auto i [[maybe_unused]] : c10::irange(niters * 2 + 100))  {
                    thread_inputs[thread_id].push_back(input_vec);
                }
                input_iters[thread_id] = 0;
            }
            std::atomic<int64_t> num_attempted_iters{0};
            std::mutex m;
            std::condition_variable worker_main_cv;
            std::condition_variable main_worker_cv;
            int64_t initialized{0};
            int64_t finished{0};
            bool start{false};
            std::vector<std::thread> callers;
            callers.reserve(ninstances);
            std::cout << "init done, benchmark start" << std::endl;
            for (const auto thread_id : c10::irange(ninstances)) {
                callers.emplace_back([&, thread_id]() {
                    // warmup 100 iters
                    for (const auto j : c10::irange(100)) {
                        (void)j;
                        runner.run(thread_inputs[thread_id][input_iters[thread_id]]);
                        ++input_iters[thread_id];
                    }
                    {
                        std::unique_lock<std::mutex> lock(m);
                        ++initialized;
                        worker_main_cv.notify_one();
                        while (!start) {
                            main_worker_cv.wait(lock);
                        }
                    }
                    while (num_attempted_iters.fetch_add(1) < total_iters) {
                        runner.run(thread_inputs[thread_id][input_iters[thread_id]]);
                        ++input_iters[thread_id];
                    }

                    {
                        std::unique_lock<std::mutex> lock(m);
                        ++finished;
                        worker_main_cv.notify_one();
                    }
                });
            }

            using Clock = std::chrono::high_resolution_clock;
            using RecordProfile = torch::autograd::profiler::RecordProfile;
            using TimePoint = std::chrono::time_point<Clock>;
            TimePoint start_time;
            {
                std::unique_lock<std::mutex> lock(m);
                while (initialized != ninstances) {
                    worker_main_cv.wait(lock);
                }
                start = true;
                start_time = Clock::now();
            }
            main_worker_cv.notify_all();
            {
                std::unique_lock<std::mutex> lock(m);
                worker_main_cv.wait(
                    lock, [&]() { return finished == ninstances; });
            }
            auto end_time = std::chrono::high_resolution_clock::now();

            float total_time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        end_time - start_time)
                                        .count() / 1000.0 / 1000.0;
            float fps = bs * ninstances * niters / total_time_ms * 1000;
            std::cout << "Throughput: " << fps << std::endl;
            for (auto& t : callers) {
                t.join();
            }
            return 0;
        }
        """
    )
    os.system(f"ln -s {tmp_dir}/model.so {target_dir}/model.so")
    os.system(f"cp {tmp_dir}/inputs.pt {target_dir}/inputs.pt")
    model_dir = f"{target_dir}/model.so"
    inputs_dir = f"{target_dir}/inputs.pt"
    src_code = inference_template % (
        ninstances,
        nbatches,
        bs,
        inputs_dir,
        model_dir,
    )
    with open(f"{target_dir}/bench.cpp", "w") as f:
        f.write(src_code)
    os.system(f"cp ./CMakeLists.txt {target_dir}/CMakeLists.txt")
    cmake_prefix_path = torch.utils.cmake_prefix_path
    pytorch_install_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_libraries = os.path.join(pytorch_install_dir, "lib")
    os.system(
        f"export CMAKE_PREFIX_PATH={cmake_prefix_path} && \
          export TORCH_LIBRARIES={torch_libraries} && cd {target_dir} && cmake . && make"
    )
    return f"{target_dir}/aoti_example"


def aot_inductor_benchmark(args, model, dtype, example_inputs):
    t = time.time()
    pid = os.getpid()
    tmp_dir = os.path.join(os.getcwd(), f"./aoti-model-{dtype}-{t}-{pid}")
    model_dir = f"{tmp_dir}/model.so"
    inputs_dir = f"{tmp_dir}/inputs.pt"
    torch._export.aot_compile(
        model, example_inputs, options={"aot_inductor.output_path": model_dir}
    )
    print(f"AOTI model saved to : {model_dir}")
    # save example inputs and loaded it in cpp later
    runner = torch._C._aoti.AOTIModelContainerRunnerCpu(model_dir, 1)  # type: ignore[call-arg]
    call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
    import torch.utils._pytree as pytree

    in_spec = pytree.treespec_loads(call_spec[0])
    from torch.export._tree_utils import reorder_kwargs

    flat_inputs = pytree.tree_flatten((example_inputs, reorder_kwargs({}, in_spec)))[0]
    flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]

    class TensorListModule(torch.nn.Module):
        def __init__(self, tensor_list):
            super(TensorListModule, self).__init__()
            self.tensor_list = tensor_list

        def forward(self):
            return self.tensor_list

    # Create an instance of the module
    module = TensorListModule(flat_inputs)
    # Save the module
    torch.jit.save(torch.jit.script(module), inputs_dir)
    print(f"example inputs saved to : {inputs_dir}")
    print(f"{tmp_dir}")
    exit()
    # gen/compile benchmark
    aoti_benchmark_compile(args, tmp_dir)


def stock_pt_optimize(args, model, optimizer, dataloader):
    example_batch = fetch_batch(dataloader)
    example_batch.sparse_features = unpack(example_batch.sparse_features)
    dense, sparse = example_batch.dense_features, example_batch.sparse_features
    autocast, dtype = parse_autocast(args.dtype)
    if args.inductor:
        from torch._inductor import config as inductor_config
        from torch._dynamo import config

        # For benchmarking and profiling performance, we set error_on_recompile=True to ensure
        # that any re-compilation triggers an error.
        # This avoids silently including compilation overhead in performance measurements.
        if not args.test_auroc:
            config.error_on_recompile = True

        inductor_config.cpp_wrapper = True
        inductor_config.cpp.enable_kernel_profile = True
        if args.inference_only:
            inductor_config.freezing = True
            if not dtype == torch.int8:
                model.eval()
            if args.dtype == "tf32":
                torch.backends.mkldnn.matmul.fp32_precision = "tf32"
        if dtype == torch.int8:
            assert args.inference_only
            from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
            import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
            from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
                X86InductorQuantizer,
            )
            from torch.export import export_for_training

            print("[Info] Running torch.compile() INT8 quantization")
            with torch.no_grad():
                example_inputs = (dense, sparse)
                exported_model = export_for_training(
                    model,
                    example_inputs,
                    strict=True,
                ).module()
                quantizer = X86InductorQuantizer()
                quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                prepared_model = prepare_pt2e(exported_model, quantizer)
                prepared_model(dense, sparse)
                converted_model = convert_pt2e(prepared_model)
                torch.ao.quantization.move_exported_model_to_eval(converted_model)
                print("[Info] Running torch.compile() with default backend")
                if args.aot_inductor:
                    aot_inductor_benchmark(
                        args,
                        converted_model,
                        torch.int8,
                        (
                            dense,
                            sparse,
                        ),
                    )
                else:
                    model(dense, sparse)
                    model = torch.compile(converted_model)
                model(dense, sparse)
                model(dense, sparse)
        else:
            with torch.no_grad(), torch.autocast("cpu", enabled=autocast, dtype=dtype):
                print("[Info] Running torch.compile() with default backend")
                if args.aot_inductor:
                    aot_inductor_benchmark(
                        args,
                        model,
                        dtype,
                        (
                            dense,
                            sparse,
                        ),
                    )
                else:
                    model(dense, sparse)
                    model = torch.compile(model)
                    model(dense, sparse)
                    model(dense, sparse)
    return model, optimizer


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop the last non-full training batch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="number of test batches",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_1t",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--interaction_branch1_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch1 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--interaction_branch2_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch2 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1696543516,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
        " That is, the dataset is kept on disk but is accessed as if it were in memory."
        " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
        " preloading the dataset when preloading takes too long or when there is "
        " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the Criteo dataset npy files.",
    )
    parser.add_argument(
        "--synthetic_multi_hot_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the MLPerf v2 synthetic multi-hot dataset npz files.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle the training set in memory. This will override mmap_mode",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.add_argument(
        "--validation_auroc",
        type=float,
        default=None,
        help="Validation AUROC threshold to stop training once reached.",
    )
    parser.add_argument(
        "--evaluate_on_epoch_end",
        action="store_true",
        help="Evaluate using validation set on each epoch end.",
    )
    parser.add_argument(
        "--evaluate_on_training_end",
        action="store_true",
        help="Evaluate using test set on training end.",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument(
        "--interaction_type",
        type=InteractionType,
        choices=list(InteractionType),
        default=InteractionType.ORIGINAL,
        help="Determine the interaction type to be used (original, dcn, or projection)"
        " default is original DLRM with pairwise dot product",
    )
    parser.add_argument(
        "--collect_multi_hot_freqs_stats",
        dest="collect_multi_hot_freqs_stats",
        action="store_true",
        help="Flag to determine whether to collect stats on freq of embedding access.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default=None,
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_distribution_type",
        type=str,
        choices=["uniform", "pareto"],
        default=None,
        help="Multi-hot distribution options.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_decay_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--print_lr",
        action="store_true",
        help="Print learning rate every iteration.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )
    parser.add_argument(
        "--print_progress",
        action="store_true",
        help="Print tqdm progress bar during training and evaluation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "bf16", "fp16", "int8", "bf32", "tf32"],
        default=None,
        help="Model dtypes.",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Whether only run inference path.",
    )
    parser.add_argument(
        "--int8-prepare",
        action="store_true",
        help="prepare int8 model weight and save to disk.",
    )
    parser.add_argument(
        "--int8-configure-dir",
        type=str,
        default="./int8_configure.json",
        help="Int8 recipe location.",
    )
    parser.add_argument(
        "--int8-model-dir",
        type=str,
        default="./dlrm-v2-int8.pt",
        help="Int8 model location.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Whether calibration only for this run.",
    )
    parser.add_argument(
        "--test_auroc",
        action="store_true",
        help="Test auroc.",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=0,
        help="log-freq to print performance statistics.",
    )
    parser.add_argument(
        "--log-freq-eval",
        type=int,
        default=0,
        help="log-freq to print performance/auroc statistics for eval.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Test auroc.",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="Test auroc.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="use same samples to benchmark",
    )
    parser.add_argument(
        "--share-weight-instance",
        type=int,
        default=0,
        help="if value > 0, will use pytorch throughputbenchmark to share weight with `value` instance",
    )
    parser.add_argument(
        "--inductor",
        action="store_true",
        help="whether use torch.compile()",
    )
    parser.add_argument(
        "--distributed-training",
        action="store_true",
        help="whether train dlrm distributed",
    )
    parser.add_argument(
        "--aot-inductor",
        action="store_true",
        help="whether use AOT Inductor path to benchmark",
    )
    parser.add_argument(
        "--cpu-lists",
        type=str,
    )
    parser.add_argument(
        "--node-id",
        type=str,
    )
    parser.add_argument(
        "--fp8",
        dest="fp8",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    return parser.parse_args(argv)


def _evaluate(
    eval_model,
    eval_dataloader: DataLoader,
    stage: str,
    epoch_num: float,
    args,
) -> float:
    """
    Evaluates model. Computes and prints AUROC. Helper function for train_val_test.

    Args:
        limit_batches (Optional[int]): Limits the dataloader to the first `limit_batches` batches.
        eval_pipeline (TrainPipelineSparseDist): pipelined model.
        eval_dataloader (DataLoader): Dataloader for either the validation set or test set.
        stage (str): "val" or "test".
        epoch_num (float): Iterations passed as epoch fraction (for logging purposes).
        print_progress (bool): Whether to print tqdm progress bar.

    Returns:
        float: auroc result
    """
    limit_batches = args.limit_val_batches
    print_progress = args.print_progress
    autocast_enabled, autocast_dtype = parse_autocast(args.dtype)
    log_freq = args.log_freq_eval
    enable_torch_profile = args.profile

    benckmark_batch = fetch_batch(eval_dataloader)
    benckmark_batch.sparse_features = unpack(benckmark_batch.sparse_features)

    def fetch_next(iterator, current_it):
        if args.benchmark:
            if current_it == limit_batches:
                raise StopIteration
            return benckmark_batch
        else:
            with record_function("generate batch"):
                next_batch = next(iterator)
            with record_function("unpack KeyJaggedTensor"):
                next_batch.sparse_features = unpack(next_batch.sparse_features)
            return next_batch

    def eval_step(model, iterator, current_it):
        batch = fetch_next(iterator, current_it)
        with record_function("model forward"):
            logits = model(batch.dense_features, batch.sparse_features)
        return logits, batch.labels

    def gather_output(label, pred):
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        if my_rank == 0:
            label_list = [torch.empty_like(label) for _ in range(my_size)]
            pred_list = [torch.empty_like(pred) for _ in range(my_size)]
        else:
            label_list = None
            pred_list = None
        dist.barrier()
        dist.gather(label, label_list, dst=0)
        dist.gather(pred, pred_list, dst=0)
        if my_rank == 0:
            return torch.cat(label_list), torch.cat(pred_list)
        else:
            return None, None

    pbar = tqdm(
        iter(int, 1),
        desc=f"Evaluating {stage} set",
        total=len(eval_dataloader),
        disable=not print_progress,
    )

    print(f"EVAL_START, EPOCH_NUM: {epoch_num}")

    if not (args.inductor and args.dtype == "int8"):
        eval_model.eval()

    device = torch.device("cpu")

    iterator = itertools.islice(iter(eval_dataloader), limit_batches)
    # Two filler batches are appended to the end of the iterator to keep the pipeline active while the
    # last two remaining batches are still in progress awaiting results.
    two_filler_batches = itertools.islice(
        iter(eval_dataloader), TRAIN_PIPELINE_STAGES - 1
    )
    iterator = itertools.chain(iterator, two_filler_batches)

    preds = []
    labels = []

    auroc_computer = metrics.AUROC(task="binary").to(device)

    it = 0
    ctx1 = torch.no_grad()
    ctx2 = torch.autocast("cpu", enabled=autocast_enabled, dtype=autocast_dtype)
    ctx3 = torch.profiler.profile(
        activities=[ProfilerActivity.CPU],
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
    )
    with ctx1, ctx2:
        if enable_torch_profile:
            p = ctx3.__enter__()
            setattr(p, "_dtype", autocast_dtype)  # noqa: B010
            setattr(p, "_mode", "eval")  # noqa: B010
        while True:
            try:
                logits, label = eval_step(eval_model, iterator, it)
                pred = torch.sigmoid(logits)
                preds.append(pred)
                labels.append(label)
                pbar.update(1)
                it += 1
            except StopIteration:
                # Dataset traversal complete
                break
        if enable_torch_profile:
            ctx3.__exit__(None, None, None)

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    if not args.distributed_training:
        num_samples = labels.shape[0]
        auroc = auroc_computer(preds.squeeze().float(), labels.float())
        print(f"AUROC over {stage} set: {auroc}.")
        print(f"Number of {stage} samples: {num_samples}")
        print(f"Final AUROC: {auroc} ")
    return auroc


def _share_weight_benchmark(
    model,
    data_loader,
    args,
):
    import contextlib
    from torch.utils import ThroughputBenchmark

    print_memory("start to init throughput benchmark")
    bench = ThroughputBenchmark(model)
    print_memory("start to add input to throughput benchmark")
    batch_count = 0
    for batch in data_loader:
        if batch_count >= args.limit_val_batches:
            break
        batch.sparse_features = unpack(batch.sparse_features)
        bench.add_input(batch.dense_features, batch.sparse_features)
        batch_count += 1
    print_memory("start to run throughput benchmark")
    ctx = contextlib.suppress()
    if args.dtype == "bf16":
        ctx = torch.autocast("cpu", enabled=True, dtype=torch.bfloat16)
    if args.dtype == "fp16":
        ctx = torch.autocast("cpu", enabled=True, dtype=torch.float16)
    with ctx:
        stats = bench.benchmark(
            num_calling_threads=args.share_weight_instance,
            num_warmup_iters=200,
            num_iters=args.limit_val_batches * args.share_weight_instance,
        )
        print(stats)
    latency = stats.latency_avg_ms
    batch_size = batch.dense_features.shape[0]
    throughput = (1 / latency) * 1000 * batch_size * args.share_weight_instance
    print("Throughput: {:.3f} fps".format(throughput))


@dataclass
class TrainValTestResults:
    val_aurocs: List[float] = field(default_factory=list)
    test_auroc: Optional[float] = None


def train_val_test(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
) -> TrainValTestResults:
    """
    Train/validation/test loop.

    Args:
        args (argparse.Namespace): parsed command line args.
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device to use.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        test_dataloader (DataLoader): Test set's dataloader.

    Returns:
        TrainValTestResults.
    """
    assert args.inference_only
    results = TrainValTestResults()

    with torch.no_grad():
        if args.share_weight_instance > 0:
            _share_weight_benchmark(model.model, val_dataloader, args)
            exit()
    # Mlperf is using val set to test auroc
    # https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/python/multihot_criteo.py#L99-L107
    test_auroc = _evaluate(
        model.model,
        val_dataloader,
        "test",
        0,
        args,
    )
    results.test_auroc = test_auroc
    return results


def construct_model(args):
    if (
        args.dtype == "int8"
        and not args.int8_prepare
        and not args.calibration
        and args.jit
    ):
        assert args.inference_only
        print(f"loading int8 model from {args.int8_model_dir}")
        model = torch.jit.load(args.int8_model_dir)
        print_memory("int8 loading finished")
        return DLRMTrain(model), None, None

    device: torch.device = torch.device("cpu")
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=(
                none_throws(args.num_embeddings_per_feature)[feature_idx]
                if args.num_embeddings is None
                else args.num_embeddings
            ),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    assert args.interaction_type == InteractionType.DCN
    from optimized_model.model import OPTIMIZED_DLRM_DCN, init_weight

    dcn_init_fn = OPTIMIZED_DLRM_DCN
    dlrm_model = dcn_init_fn(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("cpu")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=args.dense_arch_layer_sizes,
        over_arch_layer_sizes=args.over_arch_layer_sizes,
        dcn_num_layers=args.dcn_num_layers,
        dcn_low_rank_dim=args.dcn_low_rank_dim,
        dense_device=device,
    )
    if args.validation_auroc:
        # init weight to test convergence
        init_weight(dlrm_model)

    train_model = DLRMTrain(dlrm_model)
    if args.test_auroc or args.calibration:
        assert args.snapshot_dir
        print_memory("start loading checkpoint ")
        load_snapshot(train_model, args.snapshot_dir)

    from optimized_model.model import replace_crossnet

    # re-write crossnet with using nn.linear instead of only using torch.linear
    replace_crossnet(train_model.model)
    print_memory("start replace emeddingbag ")
    # embedding_optimizer = torch.optim.Adagrad if args.adagrad else torch.optim.SGD
    # This will apply the Adagrad optimizer in the backward pass for the embeddings (sparse_arch). This means that
    # the optimizer update will be applied in the backward pass, in this case through a fused op.
    # TorchRec will use the FBGEMM implementation of EXACT_ADAGRAD.
    # For GPU devices, a fused CUDA kernel is invoked. For CPU, FBGEMM_GPU invokes CPU kernels
    # https://github.com/pytorch/FBGEMM/blob/2cb8b0dff3e67f9a009c4299defbd6b99cc12b8f/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L676-L678

    # Note that lr_decay, weight_decay and initial_accumulator_value for Adagrad optimizer in FBGEMM v0.3.2
    # cannot be specified below. This equivalently means that all these parameters are hardcoded to zero.
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.adagrad:
        optimizer_kwargs["eps"] = ADAGRAD_EPS
    # apply_optimizer_in_backward(
    #     embedding_optimizer,
    #     train_model.model.sparse_arch.parameters(),
    #     optimizer_kwargs,
    # )
    model = train_model
    # model.model.sparse_arch = SparseArchTraceAbleWrapper(model.model.sparse_arch)

    # def optimizer_with_params():
    #     if args.adagrad:
    #         return lambda params: torch.optim.Adagrad(
    #             params,
    #             lr=args.learning_rate,
    #             lr_decay=ADAGRAD_LR_DECAY,
    #             weight_decay=WEIGHT_DECAY,
    #             initial_accumulator_value=ADAGRAD_INIT_ACC,
    #             eps=ADAGRAD_EPS,
    #         )
    #     else:
    #         return lambda params: torch.optim.SGD(
    #             params,
    #             lr=args.learning_rate,
    #             weight_decay=WEIGHT_DECAY,
    #         )

    # dense_optimizer = KeyedOptimizerWrapper(
    #     dict(in_backward_optimizer_filter(model.named_parameters())),
    #     optimizer_with_params(),
    # )
    # optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
    optimizer, lr_scheduler = None, None
    if not args.inference_only:
        print_memory("start create optimizer ")
        assert args.adagrad
        param = (
            list(model.model.dense_arch.parameters())
            + list(model.model.inter_arch.parameters())
            + list(model.model.over_arch.parameters())
        )
        optimizer = torch.optim.Adagrad(
            param,
            lr=args.learning_rate,
            lr_decay=ADAGRAD_LR_DECAY,
            weight_decay=WEIGHT_DECAY,
            initial_accumulator_value=ADAGRAD_INIT_ACC,
            eps=ADAGRAD_EPS,
        )
    return model, optimizer, lr_scheduler


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """

    args = parse_args(argv)
    for name, val in vars(args).items():
        try:
            vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError):
            pass

    if args.multi_hot_sizes is not None:
        if args.in_memory_binary_criteo_path is None:
            print("use dummy data")
        else:
            print("use one hot real data set to generate multi-hot data per iter")
        assert (
            args.num_embeddings_per_feature is not None
            and len(args.multi_hot_sizes) == len(args.num_embeddings_per_feature)
            or args.num_embeddings_per_feature is None
            and len(args.multi_hot_sizes) == len(DEFAULT_CAT_NAMES)
        ), "--multi_hot_sizes must be a comma delimited list the same size as the number of embedding tables."
    if args.synthetic_multi_hot_criteo_path is not None:
        print("use multi-hot real data set")
    assert (
        args.in_memory_binary_criteo_path is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--in_memory_binary_criteo_path and --synthetic_multi_hot_criteo_path are mutually exclusive CLI arguments."
    assert (
        args.multi_hot_sizes is None or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_sizes is used to convert 1-hot to multi-hot. It's inapplicable with --synthetic_multi_hot_criteo_path."
    assert (
        args.multi_hot_distribution_type is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_distribution_type is used to convert 1-hot to multi-hot. \
        It's inapplicable with --synthetic_multi_hot_criteo_path."

    backend = "gloo"
    device = torch.device("cpu")

    pprint(vars(args))
    import numpy as np

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings = None

    # Sets default limits for random dataloader iterations when left unspecified.
    if (
        args.in_memory_binary_criteo_path is None
        and args.synthetic_multi_hot_criteo_path is None
    ):
        for split in ["train", "val", "test"]:
            attr = f"limit_{split}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)

    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = args.over_arch_layer_sizes

    print_memory("start init dlrm ")
    model, optimizer, lr_scheduler = construct_model(args)
    print_memory("start create dataloader ")

    # only create 1 multi-hot sample to save memory
    if args.multi_hot_sizes is not None:
        vb, tb = args.limit_val_batches, args.limit_test_batches
        args.limit_val_batches = 1
        args.limit_test_batches = 1

    val_dataloader = get_dataloader(args, backend, "val")
    test_dataloader = get_dataloader(args, backend, "test")

    if args.multi_hot_sizes is not None:
        args.limit_val_batches, args.limit_test_batches = vb, tb

    if args.inference_only:
        train_dataloader = None
    else:
        train_dataloader = get_dataloader(args, backend, "train")

    if args.multi_hot_sizes is not None:
        print_memory("start to create Multihot")
        multihot = Multihot(
            args.multi_hot_sizes,
            args.num_embeddings_per_feature,
            args.batch_size,
            collect_freqs_stats=args.collect_multi_hot_freqs_stats,
            dist_type=args.multi_hot_distribution_type,
        )
        multihot.pause_stats_collection_during_val_and_test(model)
        print_memory("start transfer to multihot dataloader")
        if train_dataloader:
            train_dataloader = RestartableMap(
                multihot.convert_to_multi_hot, train_dataloader
            )

        val_dataloader = RestartableMap(multihot.convert_to_multi_hot, val_dataloader)
        test_dataloader = RestartableMap(multihot.convert_to_multi_hot, test_dataloader)

    if args.inductor:
        print_memory("start StockPT ")
        if args.dtype == "bf16":
            model.model.sparse_arch = model.model.sparse_arch.bfloat16()
            # model.model.inter_arch.crossnet.bias = model.model.inter_arch.crossnet.bias.bfloat16()
        if args.dtype == "fp16":
            model.model.sparse_arch = model.model.sparse_arch.half()
            # model.model.inter_arch.crossnet.bias = model.model.inter_arch.crossnet.bias.half()
        if args.fp8:
            import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer  # noqa: F401
            from utils_fp8 import inc_convert

            _, dtype = parse_autocast(args.dtype)
            inc_convert(model, dtype)
        model.model, optimizer = stock_pt_optimize(
            args, model.model, optimizer, test_dataloader
        )

    print_memory("start running model")
    train_val_test(
        args,
        model,
        optimizer,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )
    if args.collect_multi_hot_freqs_stats:
        multihot.save_freqs_stats()


if __name__ == "__main__":
    main(sys.argv[1:])
