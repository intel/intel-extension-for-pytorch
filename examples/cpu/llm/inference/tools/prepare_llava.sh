#!/usr/bin/env bash
set -e
pip install --upgrade pip
pip install --user --upgrade setuptools
llava_patch=`pwd`/llava.patch
if ! [ -f $llava_patch ]; then
    llava_patch=`pwd`/tools/llava.patch
fi
pip uninstall llava -y
rm -rf LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

pip install einops pillow sentencepiece protobuf --no-deps
git checkout intel
git apply ${llava_patch}
pip install -e . --no-deps

pip install tenacity hf_transfer lmms-eval==0.1.1 evaluate sqlitedict pycocoevalcap pycocotools --no-deps
