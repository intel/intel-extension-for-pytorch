# Use docker

* Notes

  If you use linux kernerl under version 5.4 in host, upgrade it.

* How to build an image

  Run the following commands to build a `pip` based container with the latest stable version prebuilt wheel files:

  ```console
  $ cd $DOCKERFILE_DIR
  $ DOCKER_BUILDKIT=1 docker build -f Dockerfile.prebuilt -t intel-extension-for-pytorch:prebuilt .
  $ docker run --rm intel-extension-for-pytorch:prebuilt python -c "import torch; import intel_extension_for_pytorch as ipex; print('torch:', torch.__version__,' ipex:',ipex.__version__)"
  ```

  Run the following commands to build a `conda` based container with Intel® Extension for PyTorch\* compiled from source:

  ```console
  $ cd $DOCKERFILE_DIR
  $ DOCKER_BUILDKIT=1 docker build -f Dockerfile.compile -t intel-extension-for-pytorch:compile .
  $ docker run --rm intel-extension-for-pytorch:compile python -c "import torch; import intel_extension_for_pytorch as ipex; print('torch:', torch.__version__,' ipex:',ipex.__version__)"
  ```
