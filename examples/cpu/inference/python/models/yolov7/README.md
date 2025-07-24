# YOLOv7 inference

## Model Information

| Use Case    | Framework   | Model Repository| Branch/Commit| Patch |
|-------------|-------------|-----------------|--------------|--------------|
| Inference   | Pytorch     | https://github.com/WongKinYiu/yolov7 | main/a207844 | yolov7.patch: Enable yolov7 inference with torch inductor for specified precision (fp32, int8, bf16, or fp16). |

## Preparation

* Create virtual environment `venv` and activate it:
  ```
  python3 -m venv venv
  . ./venv/bin/activate
  ```

* Install PyTorch, Torchvision
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu/
  ```

* Set IOMP and Tcmalloc Preload for better performance

  ```
  pip install packaging intel-openmp
  conda install -y gperftools -c conda-forge
  export LD_PRELOAD=<path to the intel-openmp directory>/lib/libiomp5.so:$LD_PRELOAD
  export LD_PRELOAD=<path to the tcmalloc directory>/lib/libtcmalloc.so:$LD_PRELOAD
  ```

* Download pretrained model

  Export the `CHECKPOINT_DIR` environment variable to specify the directory where the pretrained model
  will be saved. This environment variable will also be used when running quickstart scripts.
  ```
  export CHECKPOINT_DIR=<directory where the pretrained model will be saved>
  chmod a+x *.sh
  ./download_model.sh
  ```

* Prepare Dataset

  Prepare the 2017 [COCO dataset](https://cocodataset.org) for yolov7 using the `download_dataset.sh` script.
  Export the `DATASET_DIR` environment variable to specify the directory where the dataset
  will be saved. This environment variable will also be used when running quickstart scripts.
  ```
  export DATASET_DIR=<directory where the dataset will be saved>
  ./download_dataset.sh
  ```

## Inference

1. `cd examples/cpu/inference/python/models/yolov7`

2. Install general model requirements
    ```
    ./setup.sh
    ```

3. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **TEST_MODE** (THROUGHPUT, ACCURACY, REALTIME)       | `export TEST_MODE=THROUGHPUT (THROUGHPUT, ACCURACY, REALTIME)`                                   |
| **DATASET_DIR**              |                               `export DATASET_DIR=<path-to-dataset>`                                                     |
| **CHECKPOINT_DIR**      |                 `export CHECKPOINT_DIR=<directory where the pretrained model will be saved>`                                  |
| **PRECISION**    |                               `export PRECISION=fp32 <specify the precision to run: fp32, int8, bf16, or fp16>`                      |
| **OUTPUT_DIR**    |                               `export OUTPUT_DIR=<path to the directory where log files will be written>`                           |
| **MODEL_DIR** | `export MODEL_DIR=$PWD (set the current path)`                                                                                          |
| **BATCH_SIZE** (optional)  |                        `export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>`          |

4. Run `run_model.sh`

## Output
Output typically looks like this:
100%|██████████| 4/4 [00:17<00:00,  4.29s/it]
time per prompt(s): 73.46
Inference latency  73.46 ms
Throughput: 13.61 fps

Final results of the inference run can be found in `results.yaml` file.
```
results:
- key : throughput
  value: 13.61
  unit: fps
- key: latency
  value: 73.46
  unit: ms
- key: accuracy
  value: 0.20004
  unit: percentage
```
