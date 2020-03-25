#!/bin/bash

`clang-format -i torch_ipex/csrc/gpu/aten/core/*.cpp > /dev/null 2>&1`
`clang-format -i torch_ipex/csrc/gpu/aten/core/*.h > /dev/null 2>&1`
`clang-format -i torch_ipex/csrc/gpu/aten/core/*.hpp > /dev/null 2>&1`
`clang-format -i torch_ipex/csrc/gpu/aten/utils/*.cpp > /dev/null 2>&1`
`clang-format -i torch_ipex/csrc/gpu/aten/utils/*.h > /dev/null 2>&1`
`clang-format -i torch_ipex/csrc/gpu/aten/utils/*.hpp > /dev/null 2>&1`
`clang-format -i torch_ipex/csrc/gpu/aten/operators/*.cpp > /dev/null 2>&1`
`clang-format -i torch_ipex/csrc/gpu/aten/operators/*.h > /dev/null 2>&1`
`clang-format -i torch_ipex/csrc/gpu/aten/operators/*.hpp > /dev/null 2>&1`
