import torch
import torchao  # noqa: F401
from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
)
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationSpec,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)

from torch._dynamo.utils import counters
from torch._inductor.fx_passes.freezing_patterns import register_freezing_graph_pattern
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
)

quantized_decomposed = torch.ops.quantized_decomposed


def _is_valid_dqq_pattern(dtype=torch.float32):
    def _inner(match):
        assert dtype in [torch.float32, torch.bfloat16]
        q_pattern_node = match.output_node()
        dq_pattern_node = q_pattern_node.args[0]
        assert q_pattern_node.target is quantized_decomposed.quantize_per_tensor.default
        assert (
            dq_pattern_node.target is quantized_decomposed.dequantize_per_tensor.default
        )
        for i in range(2, len(q_pattern_node.args)):
            assert q_pattern_node.args[i] == dq_pattern_node.args[i]
        # TODO: reenable checking after enable cat calibration
        # assert math.isclose(
        #     q_pattern_node.args[1], dq_pattern_node.args[1], rel_tol=1e-5, abs_tol=1e-5
        # )
        cat_node = dq_pattern_node.args[0]
        return cat_node.target is torch.ops.aten.cat.default

    return _inner


def _register_dequant_quant_pass(pattern, pass_number=3, dtype=torch.float32):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dqq_pattern(dtype),
        pass_number=pass_number,
    )
    def dqq_fusion(match: Match, *args, **kwargs):
        assert dtype in [torch.float32, torch.bfloat16]

        q_pattern_node = match.output_node()
        dq_pattern_node = q_pattern_node.args[0]
        cat_node = dq_pattern_node.args[0]

        q_pattern_node.replace_all_uses_with(cat_node)
        cat_node.meta.update(q_pattern_node.meta)

        match.graph.erase_node(q_pattern_node)
        match.graph.erase_node(dq_pattern_node)

        counters["inductor"]["concat_dqq_matcher_nodes"] += 1
        counters["inductor"]["concat_dqq_matcher_nodes"] += len(match.nodes)


def _register_dqq_pattern():
    dequantize_per_tensor_activation_pattern = CallFunction(
        quantized_decomposed.dequantize_per_tensor.default,
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
    )
    quantized_op_output_pattern_pt2e = CallFunction(
        quantized_decomposed.quantize_per_tensor.default,
        dequantize_per_tensor_activation_pattern,
        KeywordArg("o_inv_scale"),
        KeywordArg("o_zp"),
        KeywordArg("o_qmin"),
        KeywordArg("o_qmax"),
        KeywordArg("o_dtype"),
    )
    print(quantized_op_output_pattern_pt2e)
    _register_dequant_quant_pass(quantized_op_output_pattern_pt2e)


def get_default_quantization_config(
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


def _calibrate(model, example_inputs):
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq  # noqa F401
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
        X86InductorQuantizer,
    )
    from torch.export import export_for_training

    exported_model = export_for_training(
        model,
        example_inputs,
        strict=True,
    ).module(check_guards=False)
    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model(*example_inputs)
    converted_model = convert_pt2e(prepared_model)
    torch.ao.quantization.move_exported_model_to_eval(converted_model)
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

    _register_dqq_pattern()

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

    model.mlps = converted_mlps
    model.dense_arch = converted_dense

    print_memory("start calibrate_embeddingbag")
    qtype = torch.int8

    class QEmbeddingBag(torch.nn.Module):
        def __init__(
            self,
            weight_shape,
            qtype,
        ):
            super().__init__()
            self.weight = None
            self.weight_scale = None

        def forward(
            self,
            input,
            offsets=None,
            per_sample_weights=None,
        ):
            # Next step: get scale through calibration rather than hardcoding
            return torch.ops.torchao._scaled_embedding_bag(
                self.weight,
                input,
                offsets,
                torch.tensor([self.weight_scale]),
                0.04366782680153847,
                0,
                True,
                torch.int8,
            )

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
            mod.weight_scale = weight_scale
            q_param = torch.clamp(
                (param / weight_scale),
                qmin,
                qmax,
            ).to(qtype)

            patched_mod = QEmbeddingBag(
                weight_shape=list(mod.weight.shape),
                qtype=qtype,
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
