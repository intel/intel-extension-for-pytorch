#!/bin/bash
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

input_path=${1}
SEQ_LEN=${SEQ_LEN:-512}
output_dir="hdf5_seq_"${SEQ_LEN}
input_file=$(basename $input_path)

python3 ./create_pretraining_data.py \
   --input_file=${input_path} \
   --output_file="${output_dir}/${input_file}" \
   --vocab_file=vocab.txt \
   --do_lower_case=True \
   --max_seq_length=$SEQ_LEN \
   --max_predictions_per_seq=76 \
   --masked_lm_prob=0.15 \
   --random_seed=12345 \
   --dupe_factor=10
