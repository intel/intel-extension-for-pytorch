# Use docker

* Notes

  If you use linux kernel under version 5.4 in host, upgrade it.

* How to build an image

  Run the following commands to build a `pip` based container with the latest stable version prebuilt wheel files:

  ```console
  $ cd $DOCKERFILE_DIR
  $ DOCKER_BUILDKIT=1 docker build -f Dockerfile.prebuilt -t intel-extension-for-pytorch:2.6.0 .
  ```

  Run the following commands to build a `conda` based container with Intel® Extension for PyTorch\* compiled from source:

  ```console
  $ git clone https://github.com/intel/intel-extension-for-pytorch.git
  $ cd intel-extension-for-pytorch
  $ git checkout v2.6.0+cpu
  $ git submodule sync
  $ git submodule update --init --recursive
  $ DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.compile -t intel-extension-for-pytorch:2.6.0 .
  ```

* Sanity Test

  When a docker image is built out, Run the command below to launch into a container:
  ```console
  $ docker run --rm -it intel-extension-for-pytorch:2.6.0 bash
  ```

  Then run the command below inside the container to verify correct installation.
  ```console
  # python -c "import torch; import intel_extension_for_pytorch as ipex; print('torch:', torch.__version__,' ipex: ',ipex.__version__)"
  ```
