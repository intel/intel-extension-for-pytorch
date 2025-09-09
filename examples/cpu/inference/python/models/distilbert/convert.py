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

import pandas as pd

tsv_file = "SST-2/dev.tsv"
csv_table = pd.read_table(tsv_file, sep="\t")
csv_table.to_csv("SST-2/dev.csv", index=False)
tsv_file = "SST-2/train.tsv"
csv_table = pd.read_table(tsv_file, sep="\t")
csv_table.to_csv("SST-2/train.csv", index=False)
