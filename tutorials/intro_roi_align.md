# Overview
The custom op, roi_align is optimized with parallelization and channels last support on the basis of the torchvision's roi_align. This guide helps you understand how to use the roi_align of IPEX in your model.

# User guide
The semantics of IPEX roi_align is exactly the same as that of torchvision. We override roi_align in the torchvision with IPEX roi_align with ATen op registration. It is activated when IPEX is imported from the Python frontend or when it is linked by a C++ program. It is totally transparent to the users.

When a torch module calls a different roi_align from that in torchvision but you are still sure the semantics is the same as that of torchvision, you can do explicit import to override the "roi_align" function or "RoIAlign" module in the code, suppose there are references to them in the module code.
```
from ... import roi_align
=> from intel_extension_for_pytorch import roi_align
```
or
```
from ... import RoIAlign
=> from intel_extension_for_pytorch import RoIAlign
```