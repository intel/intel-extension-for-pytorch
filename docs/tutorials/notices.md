Notices
=======

## Highlight

- Please take advantage of **Channels Last** memory format for image processing tasks. Comparing to PyTorch default NCHW (`torch.contiguous_format`) memory format, NHWC (`torch.channels_last`) is more friendly to Intel platforms, and thus generally yields better performance. More detailed introduction can be found at [Channels Last page](features/nhwc.html). You can get sample codes with Resnet50 at [Example page](examples.html).

## Limitations

- ``MALLOC_CONF`` setting for libjemalloc

  A recommended setting for ``MALLOC_CONF`` is ``oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000`` from performance perspective. However, in some cases the ``dirty_decay_ms:9000000000,mmuzzy_decay_ms:9000000000`` may cause Out-of-Memory crash. Please try ``oversize_threshold:1,background_thread:true,metadata_thp:auto`` instead in this case.

- `omp_set_num_threads` function failed to change OpenMP threads number of oneDNN operators if it was set before.

  `omp_set_num_threads` function is provided in Intel® Extension for PyTorch\* to change number of threads used with openmp. However, it failed to change number of OpenMP threads if it was set before.

  pseudo code:

  ```
  omp_set_num_threads(6)
  model_execution()
  omp_set_num_threads(4)
  same_model_execution_again()
  ```

  **Reason:** oneDNN primitive descriptor stores the omp number of threads. Current oneDNN integration caches the primitive descriptor in IPEX. So if we use runtime extension with oneDNN based pytorch/ipex operation, the runtime extension fails to change the used omp number of threads.

- Low performance with INT8 support for dynamic shapes

  The support for dynamic shapes in Intel® Extension for PyTorch\* INT8 integration is still working in progress. For the use cases where the input shapes are dynamic, for example inputs of variable image sizes in an object detection task or of variable sequence lengths in NLP tasks, the Intel® Extension for PyTorch\* INT8 path may slow down the model inference. In this case, please utilize stock PyTorch INT8 functionality.

- BF16 inference failed due to ``Runtime error: expected scalar type BFloat16 but found Float``.

  We are working on fixing this issue in PyTorch.
