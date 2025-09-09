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

pip install datasets evaluate
pip install scikit-learn scipy

# Check the operating system type
os_type=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
echo "My OS type: ${os_type}"

# Install model specific dependencies:
if [[ "$os_name" == *"CentOS"* ]]; then
    yum install -y git-lfs
elif [[ "$os_name" == *"Ubuntu"* || "os_name" == *"Linux"* ]]; then
    apt install -y git-lfs
fi

rm -rf transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.53.0
git lfs pull
pip install -r examples/pytorch/text-classification/requirements.txt
pip install -e ./
cd ..
