# Serving ResNet50 INT8 model with TorchServe and Intel® Extension for PyTorch optimizations

## Description
This sample provides code to integrate Intel® Extension for PyTorch with TorchServe. This project quantizes a ResNet50 model to use the INT8 Precision to improve performance on CPU.

## Preparation
You'll need to install Docker Engine on your development system. Note that while **Docker Engine** is free to use, **Docker Desktop** may require you to purchase a license. See the [Docker Engine Server installation instructions](https://docs.docker.com/engine/install/#server) for details.

## Quantize Model
Create and Quantize a TorchScript model for the INT8 precision using the python environment found in the Intel® Optimized TorchServe Container. The below command will output `rn50_int8_jit.pt` that will be used in the next step.

```bash
docker run \
    --rm -it -u root \
    --entrypoint='' \
    -v $PWD:/home/model-server \
    intel/intel-optimized-pytorch:2.2.0-serving-cpu \
    python quantize_model.py
```

> [!NOTE]
> If you are working under a corporate proxy you will need to include the following parameters in your `docker run` command: `-e http_proxy=${http_proxy} -e https_proxy=${https_proxy}`.

## Archive Model
The [Torchserve Model Archiver](https://github.com/pytorch/serve/blob/master/model-archiver/README.md) is a CLI tool found in the torchserve container as well as on [pypi](https://pypi.org/project/torch-model-archiver/). This process is very similar for the [TorchServe Workflow](https://github.com/pytorch/serve/tree/master/workflow-archiver).

Follow the instructions found in the link above depending on whether you are intending to archive a model or a workflow. Use the provided container rather than installing the archiver with the example command below:

```bash
docker run \
    --rm -it -u root \
    --entrypoint='' \
    -v $PWD:/home/model-server \
    intel/intel-optimized-pytorch:2.2.0-serving-cpu \
    torch-model-archiver \
        --model-name ipex-resnet50 \
        --version 1.0 \
        --serialized-file rn50_int8_jit.pt \
        --handler image_classifier \
        --export-path /home/model-server/model-store
```

> [!NOTE]
> If you are working under a corporate proxy you will need to include the following parameters in your `docker run` command: `-e http_proxy=${http_proxy} -e https_proxy=${https_proxy}`.

#### Advanced Model Archival
The `--handler` argument is an important component of serving as it controls the inference pipeline. Torchserve provides several default handlers [built into the application](https://pytorch.org/serve/default_handlers.html#torchserve-default-inference-handlers). that are often enough for most inference cases, but you may need to create a custom handler if your application's inference needs additional preprocessing, postprocessing or using other variables to derive a final output. 

To create a custom handler, first inherit `BaseHandler` or another built-in handler and override any necessary functionality. Usually, you only need to override the preprocessing and postprocessing methods to achieve an application's inference needs.

```python
from ts.torch_handler.base_handler import BaseHandler

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """
```

> [!TIP]
> For more examples of how to write a custom handler, see the [TorchServe documentation](https://github.com/pytorch/serve/blob/master/docs/custom_service.md).

Additionally, the `torch-model-archiver` allows you to pass additional parameters/files to tackle complex scenarios while archiving the package. 

```txt
--requirements-file     Path to requirements.txt file containing
                        a list of model specific python packages
                        to be installed by TorchServe for seamless 
                        model serving.
--extra-files           Pass comma separated path to extra dependency
                        files required for inference and can be
                        accessed in handler script.
--config-file           Path to model-config yaml files that can
                        contain information like threshold values,
                        any parameter values need to be passed from
                        training to inference.
```

> [!TIP]
> For more use-case examples, see the [TorchServe documentation](https://github.com/pytorch/serve/tree/master/examples).

## Start Server
Start the TorchServe Server.

```bash
docker run \
    -d --rm --name server \
    -v $PWD/model-store:/home/model-server/model-store \
    -v $PWD/wf-store:/home/model-server/wf-store \
    --net=host \
    intel/intel-optimized-pytorch:2.2.0-serving-cpu
```

> [!TIP]
> For more information about how to configure the TorchServe Server, see the [Intel AI Containers Documentation](https://github.com/intel/ai-containers/tree/main/pytorch/serving).

> [!NOTE]
> If you are working under a corporate proxy you will need to include the following parameters in your `docker run` command: `-e http_proxy=${http_proxy} -e https_proxy=${https_proxy}`.

Check the server logs to verify that the server has started correctly.

```bash
docker logs server
```

Register the model using the HTTP/REST API and verify it has been registered

```bash
curl -v -X POST "http://localhost:8081/models?url=ipex-resnet50.mar&initial_workers=1"
curl -v -X GET "http://localhost:8081/models"
```

Download a [test image](https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg) and make an inference request using the HTTP/REST API.

```bash
curl -v -X POST "http://localhost:8080/v2/models/ipex-resnet50/infer" \
    -T kitten_small.jpg
```

Unregister the Model

```bash
curl -v -X DELETE "http://localhost:8081/models/ipex-resnet50"
```

## Stop Server
When finished with the example, stop the torchserve server with the following command:

```bash
docker container stop server
```

## Trademark Information
Intel, the Intel logo and Intel Xeon are trademarks of Intel Corporation or its subsidiaries.
* Other names and brands may be claimed as the property of others.

&copy;Intel Corporation
