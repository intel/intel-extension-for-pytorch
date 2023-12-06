# Serving BERT models with Triton Server and Intel® Extension for PyTorch optimizations

## Description
This sample provide code to integrate Intel® Extension for PyTorch with Triton Inference Server framework. This project provides custom Python backend for Intel® Extension for PyTorch and additional dynamic batching algorithm to improve the performance. This code can be used as performance benchmark for Bert-Base and Bert-Large models.

![graph](./graph.jpg)

## Preparation
Make sure that Docker is installed on both host and client instance.
In case of running on two separate instances edit config.properties and provide required variables.
## Supported models
Currently AI Inference samples support following Bert models finetuned on Squad dataset:
- bert_base        - PyTorch+Intel® Extension for PyTorch [Bert Base uncased](https://huggingface.co/csarron/bert-base-uncased-squad-v1 "Bert Base uncased")
- bert_large        - PyTorch+Intel® Extension for PyTorch [Bert Large uncased](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad "Bert Large uncased")

## Possible run scenarios
AI Inference samples allow user to run inference on localhost or on remote Triton Server Host. 
By default config.properties is filled with localhost run option. 
### Execution on localhost
To build, start Docker containers, run tests, stop and do cleanup on localhost execute scripts in following order:

`$ bash build.sh`  - builds Docker image for Triton Server Client and Host with name specified in config.properties

`$ bash start.sh`  - runs Docker containers for Triton Server Client and Host for model specified in config.properties

`$ bash run_test.sh`  - sends requests to Triton Server Host for model specified in config.properties. Values for sequence length, number of iterations, run mode can be passed as an argument.

`$ sudo bash stop.sh`  - stops Docker containers for Triton Server Client and Host for model, and removes temporary files. 

### Execution on two separate instances

##### DISCLAIMER: This deployment is designed to be carried out on two distinct machines.
Make sure that IP address for Triton Server Host instance is provided in config.properties on instance with Triton Server Client.

Scripts to run on client Triton Server Host instance:

`$ bash build.sh host`  - builds Docker image for Triton Server Host with name specified in config.properties

`$ bash start.sh host`  - runs Docker container for localhost Triton Server Host for model specified in config.properties

`$ bash stop.sh host`  - (**run after inference is finished**) stops Docker container for Triton Server Host and removes temporary files. 

Scripts to run on client Triton Server Client instance:

`$ bash build.sh client`  - builds Docker image for Triton Server Client with name specified in config.properties

`$ bash start.sh client`  - runs Docker container for Triton Server Client for model specified in config.properties

`$ bash run_test.sh`  - sends requests to remote Triton Server Host for model specified in config.properties. Values for sequence length, number of iterations, run mode can be passed as an argument.

`$ bash stop.sh client`  - (**run after inference is finished**) stops Docker container for Triton Server Client. 

## Additional info
Downloading and loading models take some time, so please wait until you run run_test.sh script.
Model loading progress can be tracked by following Triton Server Host docker container logs.

## License 
AI Inference samples project is licensed under Apache License Version 2.0. Refer to the [LICENSE](../LICENSE) file for the full license text and copyright notice.

This distribution includes third party software governed by separate license terms.

3-clause BSD license:
- [model.py](./model_utils/bert_common/1/model.py) -  for Intel® Extension for PyTorch optimized workload

This third party software, even if included with the distribution of the Intel software, may be governed by separate license terms, including without limitation, third party license terms, other Intel software license terms, and open source software license terms. These separate license terms govern your use of the third party programs as set forth in the [THIRD-PARTY-PROGRAMS](./THIRD-PARTY-PROGRAMS) file.

## Trademark Information
Intel, the Intel logo and Intel Xeon are trademarks of Intel Corporation or its subsidiaries.
* Other names and brands may be claimed as the property of others.

&copy;Intel Corporation
