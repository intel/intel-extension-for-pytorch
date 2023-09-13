# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# https://github.com/microsoft/DeepSpeed/blob/55243f3bc8d4e751734ee2000fe3979bd4b6228c/tests/unit/utils/test_init_on_device.py

# Copyright (c) 2023 Intel Corporation
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

import torch
import intel_extension_for_pytorch as ipex
from common_utils import TestCase, TestModule
import unittest


class TestOnDevice(TestCase):
    def test_on_device(self):
        for device in ("cpu", "meta"):
            with ipex.OnDevice(dtype=torch.half, device=device):
                model = TestModule()

            for p in model.parameters():
                self.assertEqual(p.device, torch.device(device))
                self.assertEqual(p.dtype, torch.half)
                assert p.device == torch.device(device)
                assert p.dtype == torch.half


if __name__ == "__main__":
    test = unittest.main()
