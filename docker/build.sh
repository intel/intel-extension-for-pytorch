set -e
IMAGE_NAME=intel-extension-for-pytorch:gpu

docker build --build-arg http_proxy=$http_proxy \
             --build-arg https_proxy=$https_proxy \
             --build-arg no_proxy=$no_proxy \
             --build-arg UBUNTU_VERSION=20.04 \
             --build-arg PYTHON=python3.9 \
             --build-arg ICD_VER=22.28.23726.1+i419~u20.04 \
             --build-arg LEVEL_ZERO_GPU_VER=1.3.23726.1+i419~u20.04 \
             --build-arg LEVEL_ZERO_VER=1.8.1+i755~u20.04 \
             --build-arg DPCPP_VER=2022.2.0-8734 \
             --build-arg MKL_VER=2022.2.0-8748 \
             --build-arg TORCH_VERSION==1.10.0a0 \
             --build-arg IPEX_VERSION=1.10.200+gpu \
             --build-arg TORCH_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
             --build-arg IPEX_WHEEL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
             -t $IMAGE_NAME \
             -f Dockerfile.ipex-gpu .
