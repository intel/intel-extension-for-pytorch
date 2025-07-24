#
# -*- coding: utf-8 -*-
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

#

# This file is originally from: [yolov7 repo](https://github.com/WongKinYiu/yolov7/blob/main/utils/google_utils.py)

import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser(description="Download PyTorch pretrained model")
    parser.add_argument("--weight", type=str, default="yolov7.pt", help="model name")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="model path")

    args = parser.parse_args()
    file = os.path.join(args.checkpoint_dir, args.weight)
    url = f"https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{args.weight}"
    torch.hub.download_url_to_file(url, file)


if __name__ == "__main__":
    main()
