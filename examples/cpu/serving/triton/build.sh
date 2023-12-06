#!/bin/bash

# Copyright (c) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

source "$(pwd)"/config.properties

echo ""
echo "Building Triton Server container ..."
echo ""

docker build -t "${image_name}${image_tag}" -f ./Dockerfile .