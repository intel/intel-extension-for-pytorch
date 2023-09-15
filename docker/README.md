# Use docker

* Notes

  If you use linux kernerl under version 5.4 in host, upgrade it.

* How to build an image

  ```console
  $ cd $DOCKERFILE_DIR
  $ DOCKER_BUILDKIT=1 docker build -t intel-extension-for-pytorch:llm_2.2.0 .
  $ docker run --rm intel-extension-for-pytorch:compile python -c "import torch; import intel_extension_for_pytorch as ipex; print('torch:', torch.__version__,' ipex:',ipex.__version__)"
  ```
