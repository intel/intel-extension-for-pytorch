# Use docker

* Notes

  If you use linux kernerl under version 5.4 in host, upgrade it.

* How to build image

  Run the following commands to build the `Pip` based deployment container:

  ```console
  $ cd $DOCKERFILE_DIR
  $ DOCKER_BUILDKIT=1 docker build -f Dockerfile.pip -t intel-extension-for-pytorch:pip .
  $ docker run --rm intel-extension-for-pytorch:pip python -c "import torch; import intel_extension_for_pytorch as ipex; print('torch:', torch.__version__,' ipex:',ipex.__version__)"
  ```

  Run the following commands to build the `Conda` based development container:

  ```console
  $ cd $DOCKERFILE_DIR
  $ DOCKER_BUILDKIT=1 docker build -f Dockerfile.conda -t intel-extension-for-pytorch:conda .
  $ docker run --rm intel-extension-for-pytorch:conda python -c "import torch; import intel_extension_for_pytorch as ipex; print('torch:', torch.__version__,' ipex:',ipex.__version__)"
  ```
