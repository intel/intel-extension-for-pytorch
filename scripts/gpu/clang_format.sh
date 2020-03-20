#!/bin/bash

`clang-format -i torch_ipex/csrc/gpu/aten/core/*.cpp`
`clang-format -i torch_ipex/csrc/gpu/aten/core/*.h`
`clang-format -i torch_ipex/csrc/gpu/aten/core/*.hpp`
`clang-format -i torch_ipex/csrc/gpu/aten/utils/*.cpp`
`clang-format -i torch_ipex/csrc/gpu/aten/utils/*.h`
`clang-format -i torch_ipex/csrc/gpu/aten/utils/*.hpp`
`clang-format -i torch_ipex/csrc/gpu/aten/operators/*.cpp`
`clang-format -i torch_ipex/csrc/gpu/aten/operators/*.h`
`clang-format -i torch_ipex/csrc/gpu/aten/operators/*.hpp`
