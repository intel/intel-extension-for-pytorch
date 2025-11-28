#!/bin/bash

#
# Copyright (c) 2024 Intel Corporation
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

pip install datasets sentencepiece psutil lm_eval===0.4.9 peft==0.17.1

# Clone the Transformers repo in the GPT-J inference directory
git clone https://github.com/jianan-gu/transformers -b flex_attention_enabling_2.7
cd transformers
pip install -e ./
cd ..

# Get prompt.json for gneration inference
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

export EVAL_SCRIPT="run_llm_inductor_greedy.py"
export TORCH_INDUCTOR=1
