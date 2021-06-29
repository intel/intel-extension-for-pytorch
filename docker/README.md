# Use docker

* Notes

  If you use linux kernerl under version 5.4 in host, upgrade it.

* How to build image

  Run the following commands.

  ```console
  $ cd $DOCKERFILE_DIR
  $ DOCKER_BUILDKIT=1 docker build --build-arg IPEX_VERSION=v1.8.0 --build-arg PYTORCH_VERSION=v1.8.0 -t intel-extension-for-pytorch:test .
  $ docker run intel-extension-for-pytorch:test python -c "import torch;import intel_pytorch_extension as ipex;print('torch:', torch.__version__,' ipex:',ipex.__version__)"
  ```
