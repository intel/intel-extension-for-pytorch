#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
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

#get model
git lfs install
git clone https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english

#get dataset
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
unzip SST-2.zip
python convert.py
wget https://raw.githubusercontent.com/huggingface/datasets/2.0.0/metrics/accuracy/accuracy.py

#cp accuracy.py ./transformers/examples/pytorch/text-classification/
