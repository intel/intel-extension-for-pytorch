
#!/usr/bin/env bash
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
CHECKPOINT_DIR=${CHECKPOINT_DIR-$PWD}

dir=$(pwd)
mkdir -p ${CHECKPOINT_DIR}/
cd ${CHECKPOINT_DIR}/
CHECKPOINT_DIR=$(pwd)
python ${dir}/hub_help.py  --checkpoint-dir $CHECKPOINT_DIR --weight yolov7.pt
cd $dir
