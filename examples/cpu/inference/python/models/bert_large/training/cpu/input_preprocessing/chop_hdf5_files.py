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

import glob
import h5py  # F401
import numpy as np
from argparse import ArgumentParser
import os

hdf5_compression_method = None
max_pred_per_seq = 76
seq_length = 512 if "SEQ_LEN" not in os.environ else int(os.environ["SEQ_LEN"])
n_output_shards = (
    2048 if "SHARD_NUM" not in os.environ else int(os.environ["SHARD_NUM"])
)
input_path = "hdf5_seq_{}".format(seq_length)
input_files = sorted(glob.glob(input_path + "/part*", recursive=False))
print("n_input_shards =", len(input_files))

if not os.path.exists("2048_shards_uncompressed_{}".format(seq_length)):
    os.mkdir("2048_shards_uncompressed_{}".format(seq_length))

ofile_prefix = "2048_shards_uncompressed_{}/part_".format(seq_length)
ofile_suffix = "_of_" + str(n_output_shards) + ".hdf5"

print("n_output_shards =", n_output_shards)

# First pass over data to get sample count (read only the smallest array to get count)
n_samples = 0
for idx, ifile in enumerate(input_files):
    print("Scanning:", ifile, " --  Progress:", idx + 1, "/", len(input_files))
    h5_ifile = h5py.File(ifile, "r")

    f_next_sentence_labels = h5_ifile["next_sentence_labels"][:]

    h5_ifile.close()
    n_samples += f_next_sentence_labels.shape[0]


n_sample_per_ofile_nominal = (n_samples + n_output_shards - 1) // n_output_shards
n_excess = (
    n_output_shards * n_sample_per_ofile_nominal - n_samples
)  # Always a positive number

print(
    "creating ",
    n_output_shards,
    " output file handles.  This could take a while.",
    flush=True,
)
ofile_handles = [
    h5py.File(ofile_prefix + str(x) + ofile_suffix, "w") for x in range(n_output_shards)
]

ofile_idx = 0  # which output file
ofile_entry_idx = 0  # index into an individual data element of an output file
ifile_entry_idx = 0

n_samples_in_this_shard = n_sample_per_ofile_nominal - 1
o_input_ids = np.ndarray((n_samples_in_this_shard, seq_length))
o_input_masks = np.ndarray((n_samples_in_this_shard, seq_length))
o_segment_ids = np.ndarray((n_samples_in_this_shard, seq_length))
o_masked_lm_positions = np.ndarray((n_samples_in_this_shard, max_pred_per_seq))
o_masked_lm_ids = np.ndarray((n_samples_in_this_shard, max_pred_per_seq))
o_next_sentence_labels = np.ndarray((n_samples_in_this_shard))

for ifile in input_files:
    print("Processing:", ifile, " --  Progress:", idx + 1, "/", len(input_files))
    h5_ifile = h5py.File(ifile, "r")

    ifile_entry_idx = 0
    f_input_ids = h5_ifile["input_ids"][:]
    f_input_masks = h5_ifile["input_mask"][:]
    f_segment_ids = h5_ifile["segment_ids"][:]
    f_masked_lm_positions = h5_ifile["masked_lm_positions"][:]
    f_masked_lm_ids = h5_ifile["masked_lm_ids"][:]
    f_next_sentence_labels = h5_ifile["next_sentence_labels"][:]

    h5_ifile.close()

    # This could be vectorized but keeping it simple due to lack of time
    while ifile_entry_idx < f_input_ids.shape[0]:
        if ofile_entry_idx == n_samples_in_this_shard:
            ofile_handles[ofile_idx].create_dataset(
                "input_ids",
                data=o_input_ids,
                dtype="i4",
                compression=hdf5_compression_method,
            )
            ofile_handles[ofile_idx].create_dataset(
                "input_mask",
                data=o_input_masks,
                dtype="i1",
                compression=hdf5_compression_method,
            )
            ofile_handles[ofile_idx].create_dataset(
                "segment_ids",
                data=o_segment_ids,
                dtype="i1",
                compression=hdf5_compression_method,
            )
            ofile_handles[ofile_idx].create_dataset(
                "masked_lm_positions",
                data=o_masked_lm_positions,
                dtype="i4",
                compression=hdf5_compression_method,
            )
            ofile_handles[ofile_idx].create_dataset(
                "masked_lm_ids",
                data=o_masked_lm_ids,
                dtype="i4",
                compression=hdf5_compression_method,
            )
            ofile_handles[ofile_idx].create_dataset(
                "next_sentence_labels",
                data=o_next_sentence_labels,
                dtype="i1",
                compression=hdf5_compression_method,
            )
            ofile_handles[ofile_idx].flush()
            ofile_handles[ofile_idx].close()

            ofile_entry_idx = 0
            ofile_idx += 1
            print("Opening output idx:", ofile_idx)

            n_samples_in_this_shard = n_sample_per_ofile_nominal
            if ofile_entry_idx < n_excess:
                n_samples_in_this_shard -= 1

            o_input_ids = np.ndarray((n_samples_in_this_shard, seq_length))
            o_input_masks = np.ndarray((n_samples_in_this_shard, seq_length))
            o_segment_ids = np.ndarray((n_samples_in_this_shard, seq_length))
            o_masked_lm_positions = np.ndarray(
                (n_samples_in_this_shard, max_pred_per_seq)
            )
            o_masked_lm_ids = np.ndarray((n_samples_in_this_shard, max_pred_per_seq))
            o_next_sentence_labels = np.ndarray((n_samples_in_this_shard))

        o_input_ids[ofile_entry_idx] = f_input_ids[ifile_entry_idx]
        o_input_masks[ofile_entry_idx] = f_input_masks[ifile_entry_idx]
        o_segment_ids[ofile_entry_idx] = f_segment_ids[ifile_entry_idx]
        o_masked_lm_positions[ofile_entry_idx] = f_masked_lm_positions[ifile_entry_idx]
        o_masked_lm_ids[ofile_entry_idx] = f_masked_lm_ids[ifile_entry_idx]
        o_next_sentence_labels[ofile_entry_idx] = f_next_sentence_labels[
            ifile_entry_idx
        ]
        ofile_entry_idx += 1

        ifile_entry_idx += 1

if __name__ == "__main__":
    parser = ArgumentParser(description="This is a script to parse the trace file")
    parser.add_argument(
        "--trace",
        metavar="\b",
        default="test_trace_10.json",
        type=str,
        help="The trace file. ",
    )
    args = parser.parse_args()
