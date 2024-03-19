#!/usr/bin/env bash
set -e
pip uninstall llava -y
rm -rf LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

pip install einops pillow sentencepiece protobuf --no-deps
git checkout intel
git apply ../llava.patch
pip install -e . --no-deps