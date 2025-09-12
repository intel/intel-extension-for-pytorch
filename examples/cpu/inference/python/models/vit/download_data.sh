#!/bin/bash
# Copyright (c) 2024 Intel Corporation
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

DATASET_REPO="ILSVRC/imagenet-1k"
FILENAME_DATA="data/val_images.tar.gz"
OUT_DIR=${DATASET_DIR}
VAL_DIR="${OUT_DIR}/val_image"
VAL_DST="${OUT_DIR}/val"
TRAIN_DST="${OUT_DIR}/train"

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set, please create the path and set it to DATASET_DIR"
  exit 1
fi

mkdir ${OUT_DIR}
huggingface-cli download ${DATASET_REPO} ${FILENAME_DATA} --repo-type dataset --local-dir ${OUT_DIR}


val_images="${OUT_DIR}/${FILENAME_DATA}"
mkdir ${VAL_DIR}
tar -xzf ${val_images} -C ${VAL_DIR}

organize_by_classes () {
#Convert the dataset structure from
#${OUT_DIR}/val_image/ILSVRC2012_val_00000293_n01440764.JPEG to
#${OUT_DIR}/val/n01440764/ILSVRC2012_val_00000293_n01440764.JPEG
    local SRC=$1
    local DST=$2
    find "${SRC}" -type f \( -iname "*.jpeg" -o -iname "*.jpg" \) -print0 |
    while IFS= read -r -d '' file; do
        base="$(basename "$file")"
        class="${base##*_}"
        class="${class%.*}"
        if [[ ! "${class}" =~ ^n[0-9]{8}$ ]]; then
            echo "[skip] $base (no class id)"
            continue
        fi
        class_dir="${DST}/${class}"
        mkdir -p "${class_dir}"
        mv -n -- "$file" "${class_dir}/"
    done
}
organize_by_classes "${VAL_DIR}" "${VAL_DST}"
cp -r ${VAL_DST} ${TRAIN_DST}
rm -rf "${VAL_DIR}"
echo "Dataset download completed."
