Performance
===========

## Overview

This page shows performance boost with Intel® Extension for PyTorch\* on several popular topologies.

## Performance Numbers

<table border="1" cellpadding="10" align="center" class="perf_table">
<tbody>
  <col>
  <col>
  <col>
  <colgroup span="2"></colgroup>
  <colgroup span="2"></colgroup>
  <col>
  <col>
  <col>
  <tr>
    <th rowspan="2" scope="col">Hardware</th>
    <th rowspan="2" scope="col">Workload<sup>1</sup></th>
    <th rowspan="2" scope="col">Precision</th>
    <th colspan="2" scope="colgroup">Throughput Inference<sup>2</sup></th>
    <th colspan="2" scope="colgroup">Realtime Inference<sup>3</sup></th>
    <th rowspan="2" scope="col">Model Type</th>
    <th rowspan="2" scope="col">Dataset</th>
    <th rowspan="2" scope="col">Misc.</th>
  </tr>
  <tr>
    <th scope="col">Batch Size</th>
    <th scope="col">Boost Ratio</th>
    <th scope="col">Batch Size</th>
    <th scope="col">Boost Ratio</th>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" rowspan="10" scope="col">Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz</td>
    <td style="text-align: center; vertical-align: middle" scope="col">ResNet50</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">80</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.39x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.35x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Computer Vision</td>
    <td style="text-align: center; vertical-align: middle" scope="col">ImageNet</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Input shape<br />[3, 224, 224]</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">SSD-ResNet34</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">160</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.55x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.06x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Computer Vision</td>
    <td style="text-align: center; vertical-align: middle" scope="col">COCO</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Input shape<br />[3, 1200, 1200]</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">ResNext 32x16d</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">80</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.08x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.08x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Computer Vision</td>
    <td style="text-align: center; vertical-align: middle" scope="col">ImageNet</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Input shape<br />[3, 224, 224]</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">Fast R-CNN ResNet50 FPN</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">80</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.71x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.07x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Computer Vision</td>
    <td style="text-align: center; vertical-align: middle" scope="col">COCO</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Input shape<br />[3, 1200, 1200]</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">VGG-11</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">160</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.20x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.13x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Computer Vision</td>
    <td style="text-align: center; vertical-align: middle" scope="col">ImageNet</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Input shape<br />[3, 224, 224]</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">ShuffleNetv2_x1.0</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">160</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.32x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.20x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Computer Vision</td>
    <td style="text-align: center; vertical-align: middle" scope="col">ImageNet</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Input shape<br />[3, 224, 224]</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">MobileNet v2</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">160</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.48x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.12x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Computer Vision</td>
    <td style="text-align: center; vertical-align: middle" scope="col">ImageNet</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Input shape<br />[3, 224, 224]</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">DLRM</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">80</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.11x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">-</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Recommendation</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Terabyte</td>
    <td style="text-align: center; vertical-align: middle" scope="col">-</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">BERT-Large</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">80</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.14x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.02x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">NLP</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Squad</td>
    <td style="text-align: center; vertical-align: middle" scope="col">max_seq_len=384<br />Task: Question Answering</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">Bert-Base</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Float32</td>
    <td style="text-align: center; vertical-align: middle" scope="col">160</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.10x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.33x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">NLP</td>
    <td style="text-align: center; vertical-align: middle" scope="col">MRPC</td>
    <td style="text-align: center; vertical-align: middle" scope="col">max_seq_len=128<br />Task: Text Classification</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" rowspan="2" scope="col">Intel(R) Xeon(R) Platinum 8380H CPU @ 2.90GHz</td>
    <td style="text-align: center; vertical-align: middle" scope="col">BERT-Large</td>
    <td style="text-align: center; vertical-align: middle" scope="col">BFloat16</td>
    <td style="text-align: center; vertical-align: middle" scope="col">56</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.67x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.45x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">NLP</td>
    <td style="text-align: center; vertical-align: middle" scope="col">Squad</td>
    <td style="text-align: center; vertical-align: middle" scope="col">max_seq_len=384<br />Task: Question Answering</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle" scope="col">Bert-Base</td>
    <td style="text-align: center; vertical-align: middle" scope="col">BFloat16</td>
    <td style="text-align: center; vertical-align: middle" scope="col">112</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.77x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1</td>
    <td style="text-align: center; vertical-align: middle" scope="col">1.18x</td>
    <td style="text-align: center; vertical-align: middle" scope="col">NLP</td>
    <td style="text-align: center; vertical-align: middle" scope="col">MRPC</td>
    <td style="text-align: center; vertical-align: middle" scope="col">max_seq_len=128<br />Task: Text Classification</td>
  </tr>
</tbody>
</table>

<br />
<sup>1. <a href="https://github.com/IntelAI/models/tree/pytorch-r1.10-models">Model Zoo for Intel® Architecture</a></sup>
<br />
<sup>2. Throughput inference runs with single instance per socket.</sup>
<br />
<sup>3. Realtime inference runs with multiple instances, 4 cores per instance.</sup>
<br />

*Note:* Performance numbers with stock PyTorch are measured with its most performant configuration.

## Configuration

### Software Version

| Software | Version |
| :-: | :-: |
| PyTorch | [v1.10.1](https://pytorch.org/get-started/locally/) |
| Intel® Extension for PyTorch\* | [v1.10.100](https://github.com/intel/intel-extension-for-pytorch/releases) |

### Hardware Configuration

| | 3rd Generation Intel® Xeon® Scalable Processors | Products formerly Cooper Lake |
| :-: | :-: | :-: |
| CPU | Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz | Intel(R) Xeon(R) Platinum 8380H CPU @ 2.90GHz
|
| Number of nodes | 1 | 1 |
| Number of sockets | 2 | 2 |
| Cores/Socket | 40 | 28 |
| Threads/Core | 2 | 2 |
| uCode | 0xd0002a0 | 0x700001c |
| Hyper-Threading | ON | ON |
| TurboBoost | ON | ON |
| BIOS version | 04.12.02 | WLYDCRB1.SYS.0016.P29.2006080250 |
| Number of DDR Memory slots | 16 | 12 |
| Capacity of DDR memory per slot | 16GB | 64GB |
| DDR frequency | 3200 | 3200 |
| Total Memory/Node (DDR+DCPMM) | 256GB | 768GB |
| Host OS | CentOS Linux release 8.4.2105 | Ubuntu 18.04.4 LTS |
| Host Kernel | 4.18.0-305.10.2.el8\_4.x86\_64 | 4.15.0-76-generic |
| Docker OS | Ubuntu 18.04.5 LTS | Ubuntu 18.04.5 LTS |
| [Spectre-Meltdown Mitigation](https://github.com/speed47/spectre-meltdown-checker) | Mitigated | Mitigated |
