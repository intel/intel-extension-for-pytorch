set -e
IMAGE_TYPE=$1

if [ $IMAGE_TYPE == "xpu-flex" ];then

        IMAGE_NAME=intel-extension-for-pytorch:xpu-flex
        
        docker build --build-arg http_proxy=$http_proxy \
                    --build-arg https_proxy=$https_proxy \
                    --build-arg no_proxy=$no_proxy \
                    --build-arg UBUNTU_VERSION=22.04 \
                    --build-arg PYTHON=python3.10 \
                    --build-arg ICD_VER=22.43.24595.35+i538~22.04 \
                    --build-arg LEVEL_ZERO_GPU_VER=1.3.24595.35+i538~22.04 \
                    --build-arg LEVEL_ZERO_VER=1.8.8+i524~u22.04 \
                    --build-arg LEVEL_ZERO_DEV_VER=1.8.8+i524~u22.04 \
                    --build-arg DPCPP_VER=2023.0.0-25370 \
                    --build-arg MKL_VER=2023.0.0-25398 \
                    --build-arg TORCH_VERSION==1.13.0a0+gitb1dde16 \
                    --build-arg IPEX_VERSION=1.13.10+xpu \
                    --build-arg TORCHVISION_VERSION=0.14.1a0+0504df5 \
                    --build-arg TORCH_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                    --build-arg IPEX_WHEEL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                    --build-arg TORCHVISION_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                    -t $IMAGE_NAME \
                    -f Dockerfile.ipex-flex-xpu .
fi

if [ ${IMAGE_TYPE} == "xpu-max" ];then

IMAGE_NAME=intel-extension-for-pytorch:xpu-max

docker build --build-arg http_proxy=$http_proxy \
                     --build-arg https_proxy=$https_proxy \
                      --build-arg no_proxy=$no_proxy \
                      --build-arg UBUNTU_VERSION=22.04 \
                      --build-arg PYTHON=python3.10 \
                      --build-arg ICD_VER=22.43.24595.35+i538~22.04 \
                      --build-arg LEVEL_ZERO_GPU_VER=1.3.24595.35+i538~22.04 \
                      --build-arg LEVEL_ZERO_VER=1.8.8+i524~u22.04 \
                      --build-arg LEVEL_ZERO_DEV_VER=1.8.8+i524~u22.04 \
                      --build-arg DPCPP_VER=2023.0.0-25370 \
                      --build-arg MKL_VER=2023.0.0-25398  \
                      --build-arg CCL_VER=2021.8.0-25371 \
                      --build-arg TORCH_VERSION=1.13.0a0+gitb1dde16 \
                      --build-arg IPEX_VERSION=1.13.10+xpu \
                      --build-arg TORCHVISION_VERSION=0.14.1a0+0504df5 \
                      --build-arg ONECCL_BIND_PT_VERSION=1.13.100 \
                      --build-arg TORCH_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                      --build-arg IPEX_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                      --build-arg TORCHVISION_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                      --build-arg ONECCL_BIND_PT_WHL_URL=https://developer.intel.com/ipex-whl-stable-xpu \
                      -t $IMAGE_NAME \
                      -f Dockerfile.ipex-max-xpu .
fi
