import copy
import torch
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch.jit._trace import TracerWarning

from enum import IntEnum
from typing import List

import functools
import threading
import warnings
from ..utils._logger import logger, WarningType


class RunMethods(IntEnum):
    JIT = 1
    TorchDynamo = 2
    EagerInfer = 3
    EagerTrain = 4


class GraphCapture(object):
    def __init__(self, model, train, dtype, weights_prepack):
        self.model = copy.deepcopy(model)
        self.train = train
        self.dtype = dtype
        self.weights_prepack = weights_prepack
        self.method = None
        self.lock = threading.Lock()

    def __call__(self, func):
        @fake_tensor_unsupported
        def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
            try:
                with torch.no_grad():
                    traced_model = torch.jit.trace(gm.eval(), example_inputs)
                    traced_model = torch.jit.freeze(traced_model)
                return traced_model
            except Exception:
                logger.warning(
                    "JIT trace failed during the 'compiler' process.",
                    _type=WarningType.NotSupported,
                )
                return gm

        @functools.wraps(func)
        def forward(*input, **kwargs):
            if torch.jit.is_tracing():
                return func(*input, **kwargs)
            with torch.cpu.amp.autocast(
                enabled=(self.dtype == torch.bfloat16 or self.dtype == torch.half),
                dtype=self.dtype,
            ):
                if self.method:
                    if self.train:
                        return func(*input, **kwargs)
                    else:
                        return self.model(*input, **kwargs)
                else:
                    # Lock the graph generation process to avoid multiple threads generating graph simultaneously.
                    with self.lock:
                        if self.method:
                            if self.train:
                                return func(*input, **kwargs)
                            else:
                                return self.model(*input, **kwargs)
                        if self.train:
                            logger.warning(
                                "graph capture does not support training yet.",
                                _type=WarningType.NotSupported,
                            )
                            self.method = RunMethods.EagerTrain
                            return func(*input, **kwargs)
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
                                        self.model.eval(), input
                                    ).eval()
                                    traced_model = torch.jit.freeze(traced_model)
                                    output = traced_model(*input, **kwargs)
                                    self.model = traced_model
                                    self.method = RunMethods.JIT
                                    logger.debug("generate graph by JIT trace.")
                                    return output
                            except BaseException:
                                try:
                                    # JIT trace failed, try torchdynamo with JIT trace backend.
                                    torch._dynamo.reset()
                                    dynamo_model = torch._dynamo.optimize(
                                        compiler, dynamic=True
                                    )(self.model)
                                    output = dynamo_model(*input, **kwargs)
                                    self.model = dynamo_model
                                    self.method = RunMethods.TorchDynamo
                                    logger.debug("generate graph by TorchDynamo.")
                                    return output
                                except BaseException:
                                    logger.warning(
                                        "Both JIT and TorchDynamo failed, fallback to original model.",
                                        _type=WarningType.NotSupported,
                                    )
                                    self.method = RunMethods.EagerInfer
                                    torch._dynamo.reset()
                                    return self.model(*input, **kwargs)

        return forward
