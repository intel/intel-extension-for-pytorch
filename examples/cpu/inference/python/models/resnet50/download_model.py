import torch


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
hub_model_names = torch.hub.list("facebookresearch/WSL-Images")
