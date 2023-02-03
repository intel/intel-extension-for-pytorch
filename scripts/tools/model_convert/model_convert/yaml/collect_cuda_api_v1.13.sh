#!/bin/bash

pip install torch==v1.13.0
python collect_device_api.py -d cuda -f torch_cuda_api_list.yaml -s
