import warnings
import logging
from typing import Any, List
import os
import torch
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch.jit._trace import TracerWarning
from intel_extension_for_pytorch.nn.utils.model_capture import RunMethods, ModelCapture
from transformers.modeling_outputs import CausalLMOutputWithPast

jit_absolute_list = ["GPTJForCausalLM"]

num_attention_heads_list = [
    "num_decoder_attention_heads",
    "attention_heads",
    "decoder_attention_heads",
    "prompt_num_attention_heads",
    "num_cross_attention_heads",
    "encoder_attention_heads",
    "num_encoder_attention_heads",
    "num_attention_heads",
    "num_heads",
    "n_head",
    "n_heads",
]

hidden_size_list = [
    "dim",
    "embed_dim",
    "n_embd",
    "embedding_dim",
    "emb_dim",
    "projection_dim",
    "d_model",
    "true_hidden_size",
    "hidden_size",
]


class TransformersModelCapture(ModelCapture):
    def __init__(self, model, dtype, is_jit_absolute=False, weights_prepack=None):
        super().__init__(model, dtype, weights_prepack)
        self.config = model.config
        self.is_jit_absolute = (
            True
            if is_jit_absolute
            or self.model_jit_absolute()
            or os.getenv("CAPTURE_JIT_ABSOLUTE", default=None)
            else False
        )
        # self.method = RunMethods.TorchDynamoEagerInfer
        if self.train:
            self.model.train()
        else:
            self.model.eval()
        # setattr(self.model, "original_forward", self.model.forward)
        # self.convert_forward()

    # def reset_forward(self):
    #    setattr(self.model, "forward", self.model.original_forward)

    # def convert_forward(self):
    #    setattr(self.model, "forward", self.forward)

    def model_jit_absolute(self):
        if self.config.architectures[0] in jit_absolute_list:
            return True
        return False

    def jit_input_check(self, inputs, change_past_key_values):
        attr_flag = False
        num_attention_heads = 0
        hidden_size = 0
        for attr in num_attention_heads_list:
            if hasattr(self.config, attr) and isinstance(
                getattr(self.config, attr), int
            ):
                num_attention_heads = getattr(self.config, attr)
                attr_flag = True
        if not attr_flag:
            raise RuntimeError(
                "No attribute found here to represent num_attention_heads in model config,"
                "please double check model config or upstream this issue to intel pytorch team"
            )

        attr_flag = False
        for attr in hidden_size_list:
            if hasattr(self.config, attr) and isinstance(
                getattr(self.config, attr), int
            ):
                hidden_size = getattr(self.config, attr)
                attr_flag = True
        if not attr_flag:
            raise RuntimeError(
                "No attribute found here to represent hidden_size in model config,"
                "please double check model config or upstream this issue to intel pytorch team"
            )
        if change_past_key_values:
            inputs["past_key_values"] = tuple(
                [
                    (
                        torch.zeros(
                            [
                                inputs["input_ids"].size()[0],
                                num_attention_heads,
                                1,
                                hidden_size // num_attention_heads,
                            ]
                        )
                        .contiguous()
                        .to(inputs["input_ids"].device),
                        torch.zeros(
                            [
                                inputs["input_ids"].size()[0],
                                num_attention_heads,
                                1,
                                hidden_size // num_attention_heads,
                            ]
                        )
                        .contiguous()
                        .to(inputs["input_ids"].device),
                    )
                    for i in range(self.config.num_hidden_layers)
                ]
            )
            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"]
                inputs["attention_mask"] = torch.cat(
                    [
                        attention_mask.new_zeros((attention_mask.shape[0], 1)),
                        attention_mask,
                    ],
                    dim=-1,
                )
        inputs.pop("use_cache", None)
        inputs.pop("return_dict", None)
        inputs.pop("output_attentions", None)
        inputs.pop("output_hidden_states", None)
        if inputs["token_type_ids"] is None:
            inputs.pop("token_type_ids", None)
        return inputs

    def jit_output_wrapper(self, outputs, change_past_key_values):
        if change_past_key_values is True:
            past_key_values_change = tuple(
                [
                    (
                        outputs[1][i][0][:, :, 1:, :],
                        outputs[1][i][1][:, :, 1:, :],
                    )
                    for i in range(self.config.num_hidden_layers)
                ]
            )
            outputs = CausalLMOutputWithPast(
                logits=outputs[0],
                past_key_values=past_key_values_change,
            )
        else:
            outputs = CausalLMOutputWithPast(
                logits=outputs[0],
                past_key_values=outputs[1],
            )
        return outputs

    def infer(self, *args, **kwargs) -> Any:
        @fake_tensor_unsupported
        def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
            try:
                with torch.no_grad():
                    traced_model = torch.jit.trace(gm.eval(), example_inputs)
                    traced_model = torch.jit.freeze(traced_model)
                return traced_model
            except Exception:
                warnings.warn("JIT trace failed during the 'compiler' process.")
                return gm

        args = args[1]
        change_past_key_values = True if args["past_key_values"] is None else False
        if self.method == RunMethods.JITInfer:
            args = self.jit_input_check(args, change_past_key_values)
            outputs = self.model(**args)
            return self.jit_output_wrapper(outputs, change_past_key_values)

        with torch.inference_mode(), torch.no_grad(), torch.xpu.amp.autocast(
            enabled=(self.amp_dtype == torch.bfloat16 or self.amp_dtype == torch.half),
            dtype=self.amp_dtype,
        ):
            if self.method:
                return self.model(**args, **kwargs)
            else:
                # Lock the graph generation process to avoid multiple threads generating graph simultaneously.
                with self.lock:
                    if self.is_jit_absolute:
                        args = self.jit_input_check(args, change_past_key_values)
                        # self.reset_forward()
                        trace_model = torch.jit.trace(
                            self.model.eval(),
                            example_kwarg_inputs=args,
                            strict=False,
                            check_trace=True,
                        ).eval()
                        trace_model = torch.jit.freeze(trace_model)
                        outputs = trace_model(**args)
                        outputs = self.jit_output_wrapper(
                            outputs, change_past_key_values
                        )
                        torch.xpu.synchronize()
                        self.method = RunMethods.JITInfer
                        # self.convert_forward()
                        self.model = trace_model
                        logging.debug("generate graph by JIT trace.")
                    else:
                        attention_mask_change = (
                            args["attention_mask"]
                            if change_past_key_values is True
                            else None
                        )
                        use_cache_change = (
                            args["use_cache"]
                            if change_past_key_values is True
                            else None
                        )
                        token_type_ids_change = (
                            args["token_type_ids"]
                            if change_past_key_values is True
                            else None
                        )
                        return_dict_change = (
                            args["return_dict"]
                            if change_past_key_values is True
                            else None
                        )
                        output_attentions_change = (
                            args["output_attentions"]
                            if change_past_key_values is True
                            else None
                        )
                        output_hidden_states_change = (
                            args["output_hidden_states"]
                            if change_past_key_values is True
                            else None
                        )

                        try:
                            # Try JIT trace.
                            # Tracing only records operations done when the given function is run on the given
                            # tensors. Therefore, the returned ScriptModule will always run the same traced graph
                            # on any input. This has some important implications when your module is expected
                            # to run different sets of operations, depending on the input and/or the module state.
                            # In cases like these, tracing would not be appropriate, and the tracer will try to
                            # emit warnings when doing something that may cause an incorrect trace to be produced.
                            # Therefore, we catch these warnings and treat them as errors, and let TorchDynamo
                            # handle such models appropriately.
                            with warnings.catch_warnings():
                                warnings.filterwarnings("error", category=TracerWarning)
                                # self.reset_forward()
                                args = self.jit_input_check(
                                    args, change_past_key_values
                                )
                                trace_model = torch.jit.trace(
                                    self.model.eval(),
                                    example_kwarg_inputs=args,
                                    strict=False,
                                    check_trace=True,
                                ).eval()
                                trace_model = torch.jit.freeze(trace_model)
                                outputs = trace_model(**args)
                                outputs = self.jit_output_wrapper(
                                    outputs, change_past_key_values
                                )
                                torch.xpu.synchronize()
                                self.method = RunMethods.JITInfer
                                # self.convert_forward()
                                self.model = trace_model
                                logging.debug("generate graph by JIT trace.")
                        except BaseException:
                            try:
                                # JIT trace failed, try torchdynamo with JIT trace backend.
                                torch._dynamo.reset()
                                # self.reset_forward()
                                if change_past_key_values is True:
                                    args["attention_mask"] = attention_mask_change
                                    args["use_cache"] = use_cache_change
                                    args["token_type_ids"] = token_type_ids_change
                                    args["return_dict"] = return_dict_change
                                    args["output_attentions"] = output_attentions_change
                                    args[
                                        "output_hidden_states"
                                    ] = output_hidden_states_change
                                    args["past_key_values"] = None
                                # dynamo_model = torch._dynamo.optimize(
                                #    compiler, dynamic=True
                                # )(self.model)
                                dynamo_model = torch.compile(
                                    self.model, backend=compiler, dynamic=True
                                )
                                outputs = dynamo_model(**args, **kwargs)
                                torch.xpu.synchronize()
                                self.method = RunMethods.TorchDynamoEagerInfer
                                # self.convert_forward()
                                del self.model
                                self.model = dynamo_model
                                logging.debug("generate graph by TorchDynamo.")
                            except BaseException:
                                warnings.warn(
                                    "Both JIT and TorchDynamo failed, fallback to original model."
                                )
                                if change_past_key_values is True:
                                    args["attention_mask"] = attention_mask_change
                                    args["use_cache"] = use_cache_change
                                    args["token_type_ids"] = token_type_ids_change
                                    args["return_dict"] = return_dict_change
                                    args["output_attentions"] = output_attentions_change
                                    args[
                                        "output_hidden_states"
                                    ] = output_hidden_states_change
                                    args["past_key_values"] = None

                                self.method = RunMethods.EagerInfer
                                torch._dynamo.reset()
                                # self.reset_forward()
                                outputs = self.model(**args, **kwargs)
                                # self.convert_forward()
        return outputs
