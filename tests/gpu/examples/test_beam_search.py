import torch
import intel_extension_for_pytorch

input_cache = torch.tensor([[2, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64).to("xpu")
output_cache = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64).to("xpu")
beam_idx = torch.tensor([1, 2, 3, 0], dtype=torch.int64).to("xpu")
torch.ops.torch_ipex.update_beam_indices_for_cache(input_cache, output_cache,beam_idx, 4, 2, 4, 1)
print(output_cache.cpu())

