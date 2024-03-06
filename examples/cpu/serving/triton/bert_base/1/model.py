# Copyright (c) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch.utils import dlpack
from transformers import BertModel, AutoConfig
import triton_python_backend_utils as pb_utils
import intel_extension_for_pytorch as ipex
import json

def make_model(model_name, input_shape, device, bfloat16):
    print(f"{{ origin: '{model_name}', input shape: {input_shape}, enabled bfloat16: {bfloat16}}}")
    # Download PyTorch model
    config = AutoConfig.from_pretrained(
        model_name, return_dict=False, torchscript=True, num_labels=2)
    model = BertModel.from_pretrained(model_name, config=config)
    model = model.eval()
    vocab_size = model.config.vocab_size
    data = torch.randint(vocab_size, size=input_shape)

    print('Optimizing model in IPEX:')
    try:
      model = ipex.optimize(model, level="O1",auto_kernel_selection=True, conv_bn_folding=False, dtype=torch.bfloat16 if bfloat16 else torch.float32)
      with torch.no_grad(), torch.cpu.amp.autocast(enabled=bfloat16):
          model = torch.jit.trace(model, data, check_trace=False, strict=False)
          model = torch.jit.freeze(model)
    except Exception as e: print(e)
   
    print('Trigger Init Model Execution')
    # Enable fusion path (need to run forward propagation twice)
    with torch.no_grad(), torch.cpu.amp.autocast(enabled=bfloat16):
        model(data)
        model(data)
    return model.to(device)

def compute_batch_set(full_batch, batches):
    if batches is None or len(batches) == 0:
        return [full_batch,]

    batches = sorted(batches, reverse=True)
    batch_set = []
    rest_batch = full_batch
    for batch in batches:
        batch_set += [batch] * (rest_batch // batch)
        rest_batch %= batch
        if rest_batch == 0:
            break

    return batch_set

def execute_model(models, inputs, batches, dynamic_shape, bfloat16):
    input_batches = [x.shape[0] for x in inputs]

    # Join all inputs into 1 torch.Tensor
    all_inputs = torch.concat(inputs, 0)

    # Split combined inputs into batch set
    full_batch = all_inputs.shape[0]

    if not dynamic_shape:
        batches = models.keys()

    splits = compute_batch_set(full_batch, batches)
    splitted_inputs = torch.split(all_inputs, splits)

    # Execute the model
    model_outputs = []
    with torch.no_grad(), torch.cpu.amp.autocast(enabled=bfloat16):
      for i in range(len(splits)):
          inp = splitted_inputs[i]
          out = models[0 if dynamic_shape else splits[i]](inp)[1]
          model_outputs.append(out)

    # Re-combine results
    full_output = torch.concat(model_outputs, 0)
    outputs = torch.split(full_output, input_batches)

    return outputs


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        self.device = torch.device('cpu')

        # Get INPUT0 configuration
        input0_config = pb_utils.get_input_config_by_name(
            self.model_config, "INPUT0")
        seq_length = input0_config['dims'][0]

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        self.batches = []
        self.dynamic_shape = True
        self.bfloat16 = False
        parameters = self.model_config['parameters']

        if 'origin' in parameters:
          origin = parameters['origin']['string_value']
        else:
          raise pb_utils.TritonModelException("Origin model name should be defined")

        if 'batches' in parameters:
          self.batches = json.loads(parameters['batches']['string_value'])

        if 'dynamic_shape' in parameters:
          self.dynamic_shape = json.loads(parameters['dynamic_shape']['string_value'])

        if 'bfloat16' in parameters:
          self.bfloat16 = json.loads(parameters['bfloat16']['string_value'])

        self.models_cpu = dict()
        # Dynamic shapes supported in fp32/bf6 mode for PyTorch+IPEX
        if self.dynamic_shape:
          input_shape = [1, seq_length if seq_length > 0 else 128]
          self.models_cpu[0] = make_model(origin, input_shape, self.device, self.bfloat16)

        else:
          if seq_length <= 0:
            raise pb_utils.TritonModelException("Dynamic shapes switched off but input size is not defined")

          if self.batches is None or len(self.batches) == 0:
            self.batches = [1]

          for batch in self.batches:
            input_shape = [batch, seq_length]
            self.models_cpu[batch] = make_model(origin, input_shape, self.device, self.bfloat16)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Make the list of inputs in form of torch.Tensor
        inputs = []
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").to_dlpack()
            in_0_cpu = dlpack.from_dlpack(in_0).to(self.device)
            inputs.append(in_0_cpu)

        outputs = execute_model(self.models_cpu, inputs, self.batches, self.dynamic_shape, self.bfloat16)

        # Convert model outputs to triton responses
        responses = []
        for cur_bert_output in outputs:
            pooler_output = cur_bert_output.cpu().detach().numpy()
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", pooler_output.astype(self.output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')