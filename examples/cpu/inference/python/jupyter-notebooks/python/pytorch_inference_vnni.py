#!/usr/bin/env python
# encoding: utf-8

"""
==============================================================
 Copyright © 2023 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
"""

import os
from time import time
import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from torchvision import models
from transformers import BertModel

SUPPORTED_MODELS = ["resnet50", "bert"]  # models supported by this code sample

# ResNet sample data parameters
RESNET_BATCH_SIZE = 64

# BERT sample data parameters
BERT_BATCH_SIZE = 64
BERT_SEQ_LENGTH = 512

os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_VNNI"

"""
Function to perform inference on Resnet50 and BERT
"""


def runInference(model, data, modelName="resnet50", dataType="FP32", amx=True):
    """
    Input parameters
        model: the PyTorch model object used for inference
        data: a sample input into the model
        modelName: str representing the name of the model, supported values - resnet50, bert
        dataType: str representing the data type for model parameters, supported values - FP32, BF16, INT8
        amx: set to False to disable AMX on BF16, Default: True
    Return value
        inference_time: the time in seconds it takes to perform inference with the model
    """

    # Display run case
    if amx:
        isa_text = "AVX512_CORE_AMX"
    else:
        isa_text = "AVX512_CORE_VNNI"
    print("%s %s inference with %s" % (modelName, dataType, isa_text))

    # Special variables for specific models
    batch_size = None
    if "resnet50" == modelName:
        batch_size = RESNET_BATCH_SIZE
    elif "bert" == modelName:
        d = torch.randint(
            model.config.vocab_size, size=[BERT_BATCH_SIZE, BERT_SEQ_LENGTH]
        )  # sample data input for torchscript and inference
        batch_size = BERT_BATCH_SIZE
    else:
        raise Exception(
            "ERROR: modelName %s is not supported. Choose from %s"
            % (modelName, SUPPORTED_MODELS)
        )

    # Prepare model for inference based on precision (FP32, BF16, INT8)
    if "INT8" == dataType:
        # Quantize model to INT8 if needed (one time)
        model_filename = "quantized_model_%s.pt" % modelName
        if not os.path.exists(model_filename):
            qconfig = ipex.quantization.default_static_qconfig
            prepared_model = prepare(model, qconfig, example_inputs=data, inplace=False)
            converted_model = convert(prepared_model)
            with torch.no_grad():
                if "resnet50" == modelName:
                    traced_model = torch.jit.trace(converted_model, data)
                elif "bert" == modelName:
                    traced_model = torch.jit.trace(
                        converted_model, (d,), check_trace=False, strict=False
                    )
                else:
                    raise Exception(
                        "ERROR: modelName %s is not supported. Choose from %s"
                        % (modelName, SUPPORTED_MODELS)
                    )
                traced_model = torch.jit.freeze(traced_model)
            traced_model.save(model_filename)

        # Load INT8 model for inference
        model = torch.jit.load(model_filename)
        model.eval()
        model = torch.jit.freeze(model)
    elif "BF16" == dataType:
        model = ipex.optimize(model, dtype=torch.bfloat16)
        with torch.no_grad():
            with torch.cpu.amp.autocast():
                if "resnet50" == modelName:
                    model = torch.jit.trace(model, data)
                elif "bert" == modelName:
                    model = torch.jit.trace(
                        model, (d,), check_trace=False, strict=False
                    )
                else:
                    raise Exception(
                        "ERROR: modelName %s is not supported. Choose from %s"
                        % (modelName, SUPPORTED_MODELS)
                    )
                model = torch.jit.freeze(model)
    else:  # FP32
        with torch.no_grad():
            if "resnet50" == modelName:
                model = torch.jit.trace(model, data)
            elif "bert" == modelName:
                model = torch.jit.trace(model, (d,), check_trace=False, strict=False)
            else:
                raise Exception(
                    "ERROR: modelName %s is not supported. Choose from %s"
                    % (modelName, SUPPORTED_MODELS)
                )
            model = torch.jit.freeze(model)

    # Run inference
    with torch.no_grad():
        if "BF16" == dataType:
            with torch.cpu.amp.autocast():
                # Warm up
                for i in range(5):
                    model(data)

                # Measure latency
                start_time = time()
                model(data)
                end_time = time()
        else:
            # Warm up
            for i in range(5):
                model(data)

            # Measure latency
            start_time = time()
            model(data)
            end_time = time()
    inference_time = end_time - start_time
    print("Inference on batch size %d took %.3f seconds" % (batch_size, inference_time))

    return inference_time


"""
Perform all types of inference in main function

Inference run cases for both Resnet50 and BERT
1) INT8 using AVX512_CORE_VNNI
"""


def main():
    # ResNet50
    resnet_model = models.resnet50(pretrained=True)
    resnet_data = torch.rand(RESNET_BATCH_SIZE, 3, 224, 224)
    resnet_model.eval()
    int8_with_vnni_resnet_inference_time = runInference(
        resnet_model, resnet_data, modelName="resnet50", dataType="INT8", amx=False
    )

    # BERT
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    # torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    bert_data = torch.randint(
        bert_model.config.vocab_size, size=[BERT_BATCH_SIZE, BERT_SEQ_LENGTH]
    )
    bert_model.eval()
    int8_with_vnni_bert_inference_time = runInference(
        bert_model, bert_data, modelName="bert", dataType="INT8", amx=False
    )


if __name__ == "__main__":
    main()
