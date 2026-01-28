import torch
from torch.nn import functional as F
import torchao  # noqa: F401
from torchao.quantization.pt2e.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantizer.quantizer import (
    QuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.utils import (
    QuantizationConfig,
)


quantized_decomposed = torch.ops.quantized_decomposed


def get_dlrm_quantization_config(
    is_qat: bool = False,
    is_dynamic: bool = False,
    reduce_range: bool = False,
):
    """
    reduce_range is False by default. Set it to True on earlier CPUs without VNNI to avoid accuracy issue.
    """
    extra_args = {"eps": 2**-12}
    act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

    # Copy from x86 default qconfig from torch/ao/quantization/qconfig.py
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=is_dynamic,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )

    weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )
    bias_quantization_spec = None
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_qat,
    )
    return quantization_config


def print_memory(stage):
    import os
    import psutil
    import time

    print(
        f"calibrate-memory-usage-log: {time.time()}, {stage}, {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024}"
    )


def _get_first_scale(model):
    for node in model.graph.nodes:
        if "quantize_per_tensor_default" == str(node):
            return node.args[1]
    assert 0


def _calibrate(model, example_inputs):
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq  # noqa F401
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
        X86InductorQuantizer,
    )
    from torch.export import export

    exported_model = export(
        model,
        example_inputs,
        strict=True,
    ).module(check_guards=False)
    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_dlrm_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model(*example_inputs)
    converted_model = convert_pt2e(prepared_model)
    torchao.quantization.pt2e.move_exported_model_to_eval(converted_model)
    return converted_model


class MLPs(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inter_arch = model.inter_arch
        self.over_arch = model.over_arch

    def forward(self, concat_sparse_dense):
        concatenated_dense = self.inter_arch(concat_sparse_dense)
        logits = self.over_arch(concatenated_dense)
        return logits


class DLRMInt8Inference(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mlps = MLPs(model)
        self.sparse_arch = model.sparse_arch
        self.dense_arch = model.dense_arch
        del model.sparse_arch
        del model.dense_arch
        del model.inter_arch
        del model.over_arch

    def forward(self, dense_features, sparse_features):
        embedded_dense = self.dense_arch(dense_features)
        concat_sparse_dense = self.sparse_arch(embedded_dense, sparse_features)
        logits = self.mlps(concat_sparse_dense)
        return logits


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


def calibrate(model, dense, sparse):

    def get_embedded_concat(model, dense, sparse):
        embedded_dense = model.dense_arch(dense)
        embedded_concat = model.sparse_arch(embedded_dense, sparse)
        return embedded_concat.float()

    embedded_concat = get_embedded_concat(model, dense, sparse)
    print_memory("before calibrate")
    model = DLRMInt8Inference(model)

    print_memory("start calibrate mlp")
    converted_dense = _calibrate(model.dense_arch, (dense,))
    converted_mlps = _calibrate(model.mlps, (embedded_concat,))
    mlps_scale = _get_first_scale(converted_mlps)
    model.sparse_arch.scale = mlps_scale

    model.mlps = converted_mlps
    model.dense_arch = converted_dense

    print_memory("start calibrate_embeddingbag")
    qtype = torch.int8

    class QEmbeddingBag(torch.nn.Module):
        def __init__(
            self,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            mode,
            sparse,
            include_last_offset,
            padding_idx,
        ):
            super().__init__()
            self.weight = None
            self.weight_scale = None
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.mode = mode
            self.sparse = sparse
            self.include_last_offset = include_last_offset
            self.padding_idx = padding_idx

        def forward(
            self,
            input,
            offsets=None,
            per_sample_weights=None,
        ):
            weight = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                self.weight.data,
                self.weight_scale,
                0,
                -128,
                127,
                torch.int8,
            )

            res = F.embedding_bag(
                input,
                weight,
                offsets,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.mode,
                self.sparse,
                per_sample_weights,
                self.include_last_offset,
                self.padding_idx,
            )
            return res

    if qtype == torch.int8:
        qmin = torch.iinfo(qtype).min
        qmax = torch.iinfo(qtype).max
    elif qtype == torch.float8_e4m3fn:
        qmin = torch.finfo(qtype).min
        qmax = torch.finfo(qtype).max

    table_size = [
        (
            getattr(
                model.sparse_arch.embedding_bag_collection.embedding_bags, f"t_cat_{i}"
            ).weight.shape[0],
            i,
        )
        for i in range(26)
    ]
    table_size.sort(key=lambda x: x[0])
    table_sorted = [x[1] for x in table_size]
    with torch.no_grad():
        for i in table_sorted:
            name = f"t_cat_{i}"
            mod = getattr(
                model.sparse_arch.embedding_bag_collection.embedding_bags, name
            )
            mod_type_str = mod.__class__.__name__
            print(mod_type_str, name)
            param = mod.weight
            xmax = torch.max(torch.abs(param))
            weight_scale = xmax / qmax
            q_param = torch.clamp(
                (param / weight_scale),
                qmin,
                qmax,
            ).to(qtype)

            patched_mod = QEmbeddingBag(
                max_norm=mod.max_norm,
                norm_type=mod.norm_type,
                scale_grad_by_freq=mod.scale_grad_by_freq,
                mode=mod.mode,
                sparse=mod.sparse,
                include_last_offset=mod.include_last_offset,
                padding_idx=mod.padding_idx,
            )
            patched_mod.weight_scale = weight_scale.item()
            patched_mod.weight = q_param

            setattr(
                model.sparse_arch.embedding_bag_collection.embedding_bags,
                name,
                patched_mod,
            )
            print_memory(f"{name} end")

    return model
