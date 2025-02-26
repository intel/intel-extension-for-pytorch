# Intel® Extension for PyTorch\* Large Language Model (LLM) Feature Get Started For Phi-4 models

Intel® Extension for PyTorch\* provides dedicated optimization for running Phi-4 models faster, including technical points like paged attention, ROPE fusion, etc. And a set of data types are supported for various scenarios, including BF16, Weight Only Quantization INT4 (prototype), etc.

# 1. Environment Setup

There are several environment setup methodologies provided.
You can choose either of them according to your usage scenario.
The Docker-based one is recommended.

## 1.1 [RECOMMENDED] Docker-based environment setup

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 2.7-pre-phi-4
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile
# To have a custom ssh server port for multi-nodes run, please add --build-arg PORT_SSH=<CUSTOM_PORT> ex: 2345, otherwise use the default 22 SSH port
docker build -f examples/cpu/llm/Dockerfile --build-arg PORT_SSH=2345 -t ipex-llm:2.7.0-pre-phi-4 .

# Run the container with command below
docker run --rm -it --privileged -v /dev/shm:/dev/shm ipex-llm:2.7.0-pre-phi-4 bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables for inference
source ./tools/env_activate.sh inference
```

## 1.2 Conda-based environment setup

```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout 2.7-pre-phi-4
git submodule sync
git submodule update --init --recursive

# Create a conda environment (pre-built wheel only available with python=3.10)
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
cd examples/cpu/llm
bash ./tools/env_setup.sh 15

# Activate environment variables for infernece
source ./tools/env_activate.sh inference
```

<br>

# 2. How To Run Phi-4 models with ipex.llm

**ipex.llm provides a single script to facilitate running generation tasks as below:**

```
python run.py --help # for more detailed usages
```

| Key args of run.py | Notes |
|---|---|
| model id | "--model-name-or-path" or "-m" to specify the <PHI4_MODEL_ID_OR_LOCAL_PATH>, it is model id from Huggingface or downloaded local path |
| generation | default: beam search (beam size = 4), "--greedy" for greedy search |
| input tokens or prompt | provide fixed sizes for input prompt size, use "--input-tokens" for <INPUT_LENGTH> in [1024, 2048, 4096, 8192, 32768, 130944]; if "--input-tokens" is not used, use "--prompt" to choose other strings as inputs|
| input images | default: None, use "--image-url" to choose the image link address for vision-text tasks |
| output tokens | default: 32, use "--max-new-tokens" to choose any other size |
| batch size |  default: 1, use "--batch-size" to choose any other size |
| token latency |  enable "--token-latency" to print out the first or next token latency |
| generation iterations | use "--num-iter" and "--num-warmup" to control the repeated iterations of generation, default: 100-iter/10-warmup |
| streaming mode output | greedy search only (work with "--greedy"), use "--streaming" to enable the streaming generation output |
| input mode | default: 0, use "--input-mode" to choose input mode for multimodal models. 0: language; 1: vision; 2: speech; 3: vision and speech |
| input audios | default: None, use "--audio" to choose the audio link address for speech tasks |

**Note**: [_numactl_](https://linux.die.net/man/8/numactl) is used to specify memory and cores of your hardware to get better performance.
_\<-m\>_ specifies the [numa](https://en.wikipedia.org/wiki/Non-uniform_memory_access) node id (e.g., 0 to use the memory from the first numa node).
_\<-C\>_ specifies phsysical cores which you are using from the specified numa node.

The following `OMP_NUM_THREADS` and `numactl` settings in the example commands are based on the assumption that
the target server has 2 numa sockets with 56 physical cores for each numa socket.
Please adjust the configurations per your hardware.
You can use [_lscpu_](https://man7.org/linux/man-pages/man1/lscpu.1.html) command in Linux to check the numa node information.

## 2.1 Example commands for running Phi-4-mini in BF16

The <PHI_4_MINI_MODEL_ID_OR_LOCAL_PATH> in the below command specifies the Phi-4-mini model you will run.
The model can be found in [HuggingFace Models](https://huggingface.co/models).

```bash
export model=<PHI_4_MINI_MODEL_ID_OR_LOCAL_PATH>
export input=1024
export output=1024
```

**Single instance**

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m $model --input-tokens $input --max-new-tokens $output --dtype bfloat16 --ipex --token-latency --greedy
```

**Multi instances**

To benchmark 2 instances of the phi-4-mini model with each instance run on one numa socket:

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 python run.py --benchmark -m $model --input-tokens $input --max-new-tokens $output --dtype bfloat16 --ipex --token-latency --greedy &
OMP_NUM_THREADS=56 numactl -m 1 -C 56-111 python run.py --benchmark -m $model --input-tokens $input --max-new-tokens $output --dtype bfloat16 --ipex --token-latency --greedy
```

To benchmark 2 instances of the phi-4-mini model on one numa socket:

```bash
OMP_NUM_THREADS=28 numactl -m 0 -C 0-27 python run.py --benchmark -m $model --input-tokens $input --max-new-tokens $output --dtype bfloat16 --ipex --token-latency --greedy &
OMP_NUM_THREADS=28 numactl -m 0 -C 28-55 python run.py --benchmark -m $model --input-tokens $input --max-new-tokens $output --dtype bfloat16 --ipex --token-latency --greedy
```

## 2.2 Example commands for running Phi-4-multimodal in BF16

The <PHI_4_MULTIMODAL_MODEL_ID_OR_LOCAL_PATH> in the below command specifies the Phi-4-multimodal model you will run.
The model can be found in [HuggingFace Models](https://huggingface.co/models).

```bash
export KMP_AFFINITY=granularity=fine,compact,1,0
export model=<PHI_4_MULTIMODAL_MODEL_ID_OR_LOCAL_PATH>
export output=1024

# Get the example image and audio files
wget https://www.ilankelman.org/stopsigns/australia.jpg
wget https://voiceage.com/wbsamples/in_mono/Trailer.wav
```

**Test with pure text**

```bash
OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 python run.py --input-mode 0 --benchmark -m $model --ipex --token-latency --greedy --dtype bfloat16 --max-new-tokens $output --prompt "<|system|>You are a helpful assistant.<|end|><|user|>How to explain Internet for a medieval knight?<|end|><|assistant|>"
```

**Test with image and text**

```bash
OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 python run.py --input-mode 1 --benchmark -m $model --ipex --token-latency --greedy --dtype bfloat16 --max-new-tokens $output --prompt "<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>" --image-url australia.jpg
```

**Test with speech and text**

```bash
OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 python run.py --input-mode 2 --benchmark -m $model --ipex --token-latency --greedy --dtype bfloat16 --max-new-tokens $output --prompt "<|user|><|audio_1|>Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the original transcript and the translation.<|end|><|assistant|>" --audio Trailer.wav
```

**Test with image and speech**

```bash
OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 python run.py --input-mode 3 --benchmark -m $model --ipex --token-latency --greedy --dtype bfloat16 --max-new-tokens $output --prompt "<|user|><|image_1|><|audio_1|><|end|><|assistant|>" --audio Trailer.wav --image-url australia.jpg
```

<br>

# 3. Miscellaneous Tips

Intel® Extension for PyTorch\* also provides dedicated optimization for many other Large Language Models (LLM), which cover a set of data types that are supported for various scenarios.
For more details, please check [the Intel® Extension for PyTorch\* regular release doc](https://github.com/intel/intel-extension-for-pytorch/blob/v2.6.0%2Bcpu/examples/cpu/llm/README.md).
