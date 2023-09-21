import copy
import torch
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch.jit._trace import TracerWarning

from enum import IntEnum
from typing import List, Any

import functools
import logging
import threading
import warnings


class RunMethods(IntEnum):
    EagerInfer = 11
    JITInfer = 12
    TorchDynamoEagerInfer = 13
    TorchDynamoInductorInfer = 14
    EagerTrain = 111


class ModelCapture(torch.nn.Module):
    def __init__(self, model, dtype, weights_prepack, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.train = None
        self.amp_dtype = dtype
        self.weights_prepack = weights_prepack
        self.method = None
        self.lock = threading.Lock()
        self.is_jit_absolute = False

    #def __call__(self, *args: Any, **kwargs: Any) -> Any:
    #    return self.forward(args, kwargs)

    def is_jit_mode(self):
        return self.method == RunMethods.JITInfer

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

        if self.method == RunMethods.JITInfer:
            return self.model(**args)
        with torch.xpu.amp.autocast(
            enabled=(self.amp_dtype == torch.bfloat16 or self.amp_dtype == torch.half),
            dtype=self.amp_dtype,
        ):
            if self.method:
                return self.model(*args, **kwargs)
            else:
                # Lock the graph generation process to avoid multiple threads generating graph simultaneously.
                with self.lock:
                    if self.is_jit_absolute:
                        self.model = torch.jit.trace(
                            self.model.eval(), args
                        ).eval()
                        self.model = torch.jit.freeze(self.model)
                        output = self.model(*args, **kwargs)
                        self.method = RunMethods.JITInfer
                        logging.debug("generate graph by JIT trace.")
                    else:
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
                                warnings.filterwarnings(
                                    "error", category=TracerWarning
                                )
                                traced_model = torch.jit.trace(
                                    self.model.eval(), args
                                ).eval()
                                traced_model = torch.jit.freeze(traced_model)
                                output = traced_model(*args, **kwargs)
                                self.model = traced_model
                                self.method = RunMethods.JITInfer
                                logging.debug("generate graph by JIT trace.")
                        except BaseException:
                            try:
                                # JIT trace failed, try torchdynamo with JIT trace backend.
                                torch._dynamo.reset()
                                dynamo_model = torch._dynamo.optimize(
                                    compiler, dynamic=True
                                )(self.model)
                                output = dynamo_model(*args, **kwargs)
                                self.model = dynamo_model
                                self.method = RunMethods.TorchDynamoEagerInfer
                                logging.debug("generate graph by TorchDynamo.")
                            except BaseException:
                                warnings.warn(
                                    "Both JIT and TorchDynamo failed, fallback to original model."
                                )
                                self.method = RunMethods.EagerInfer
                                torch._dynamo.reset()
                                output = self.model(*args, **kwargs)

        return output

    def forward(self, *args, **kwargs) -> Any:
        if not self.train:
            return self.infer(args, kwargs)
        else:
            return None

