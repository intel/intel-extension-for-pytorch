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

### This file is originally from: https://github.com/WongKinYiu/yolov7/blob/main/scripts/get_coco.sh

DATASET_DIR=${DATASET_DIR-$PWD}

d=$DATASET_DIR
pdir=$(pwd)

# Download/unzip labels
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco2017labels-segments.zip' # or 'coco2017labels.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

# Download/unzip images
d=$DATASET_DIR/coco/images  # unzip directory 
url=http://images.cocodataset.org/zips/
f='val2017.zip'   # 1G, 5k images

echo 'Downloading' $url$f '...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

wait # finish background tasks

cd $pdir
