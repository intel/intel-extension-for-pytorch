#!/usr/bin/env bash
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

pip install tensorboard
pip install protobuf==3.20.3 numpy==1.23.5 pillow==10.3.0

# Check the operating system type
os_type=$(awk -F= '/^NAME/{print $2}' /etc/os-release)

# Install model specific dependencies:
if [[ "$os_name" == *"CentOS"* ]]; then
    yum install -y git-lfs
elif [[ "$os_name" == *"Ubuntu"* ]]; then
    apt install -y git-lfs
fi

# install torchao from source
rm -rf ao
git clone https://github.com/pytorch/ao.git
cd ao
USE_CPU_KERNELS=1 python setup.py install
cd ..

rm -rf transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.38.1
git lfs pull
git apply ../../transformers.diff
pip install -e ./
cd ..
