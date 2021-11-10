Notices
=======

## Highlight

- Please take advantage of **Channels Last** memory format for image processing tasks. Comparing to PyTorch default NCHW (`torch.contiguous_format`) memory format, NHWC (`torch.channels_last`) is more friendly to Intel platforms, and thus generally yields better performance. More detailed introduction can be found at [Channels Last page](features/nhwc.html). You can get sample codes with Resnet50 at [Example page](examples.html).

## Known Issues

- ``MALLOC_CONF`` setting for libjemalloc

  A recommended setting for ``MALLOC_CONF`` is ``oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000``. However, in some cases the ``dirty_decay_ms:9000000000,mmuzzy_decay_ms:9000000000`` may cause some performance issue. Please try ``oversize_threshold:1,background_thread:true,metadata_thp:auto`` instead in this case.

- `omp_set_num_threads` function failed to change OpenMP threads number if it was set before.

  `omp_set_num_threads` function is provided in Intel® Extension for PyTorch\* to change number of threads used with openmp. However, it failed to change number of OpenMP threads if it was set before.

  pseudo code:

  ```
  omp_set_num_threads(6)
  model_execution()
  omp_set_num_threads(4)
  same_model_execution_again()
  ```

  **Reason:** oneDNN primitive cache will store the omp number of threads. So if we use runtime extension with oneDNN based pytorch/ipex operation, the runtime extension fails to change the used omp number of threads.
