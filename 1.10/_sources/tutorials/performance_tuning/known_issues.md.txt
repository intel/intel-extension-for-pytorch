Known Issues
============

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

- Low throughtput with DLRN FP32 Train

  A 'Sparse Add' [PR](https://github.com/pytorch/pytorch/pull/23057) is pending on review. The issue will be fixed when the PR is merged.
