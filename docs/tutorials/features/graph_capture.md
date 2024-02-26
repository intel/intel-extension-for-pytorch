Graph Capture (Prototype)
============================

### Feature Description

This feature automatically applies a combination of TorchScript trace technique and TorchDynamo to try to generate a graph model, for providing a good user experience while keeping execution fast. Specifically, the process tries to generate a graph with TorchScript trace functionality first. In case of generation failure or incorrect results detected, it changes to TorchDynamo with TorchScript backend. Failure of the graph generation with TorchDynamo triggers a warning message. Meanwhile the generated graph model falls back to the original one. I.e. the inference workload runs in eager mode. Users can take advantage of this feature through a new knob `--graph_mode` of the `ipex.optimize()` function to automatically run into graph mode.

### Usage Example

[//]: # (marker_feature_graph_capture)
[//]: # (marker_feature_graph_capture)
