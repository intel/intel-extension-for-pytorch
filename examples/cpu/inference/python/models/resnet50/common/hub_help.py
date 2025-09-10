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
# ****************************************************************************

import torch.hub
from torch.hub import load_state_dict_from_url
import argparse


parser = argparse.ArgumentParser(description="Download PyTorch pretrained model")

parser.add_argument(
    "-u",
    "--url",
    type=str,
    metavar="URL",
    help="url for pretrained checkpoint your download",
)


def main():
    args = parser.parse_args()

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    hub_model_names = torch.hub.list("facebookresearch/WSL-Images")

    state_dict = load_state_dict_from_url(args.url, progress=True)


if __name__ == "__main__":
    main()
