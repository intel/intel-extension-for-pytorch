# Use docker

* Notes

If you use linux kernerl under version 5.4 in host, upgrade it.

* How to build image

Run the following commands.

```console
$ cd $DOCKERFILE_DIR
$ DOCKER_BUILDKIT=1 docker build -t intel-extension-for-pytorch:test .
```