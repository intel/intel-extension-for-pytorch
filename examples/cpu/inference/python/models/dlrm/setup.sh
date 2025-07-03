#!/bin/bash
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

pip install torchrec --index-url https://download.pytorch.org/whl/cpu --no-deps
pip install torchmetrics==1.0.3
pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cpu
pip install torchsnapshot
pip install pyre_extensions
pip install iopath
