#!/usr/bin/env bash
#
# Copyright (c) 2023-2024 Intel Corporation
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

# Install dependency
pip install tensorboardX
pip install datasets==1.11.0 accelerate tfrecord intel-openmp faiss-cpu tfrecord
pip install h5py
pip install --upgrade huggingface_hub
pip install tensorflow-cpu protobuf==3.20.3 numpy==1.20

# Check the operating system type
os_type=$(awk -F= '/^NAME/{print $2}' /etc/os-release)

# Install model specific dependencies:
if [[ "$os_name" == *"CentOS"* ]]; then
    yum install -y git-lfs
elif [[ "$os_name" == *"Ubuntu"* ]]; then
    apt install -y git-lfs
fi

rm -rf transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.38.1
git lfs pull
git apply ../../common/enable_ipex_for_transformers.diff
pip install -e ./
cd ..
