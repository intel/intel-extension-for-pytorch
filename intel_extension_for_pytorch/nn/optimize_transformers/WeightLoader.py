import torch
import torch.nn as nn
import tqdm
import json
import torch.distributed as dist
import re

class WeightLoader:
    def __init__(self, ckpt_file, tp_group=None, tp_size=0, device="xpu"):
        # self.ckpt_file = ckpt_file
        self.ckpt_dict = self.load_checkpoint(ckpt_file)
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.device = device
        self.weight_embedding = None
        if self.tp_group is not None:
            self.gpu_idex = dist.get_rank(group=tp_group)
        else:
            self.gpu_idex = 0

    def load_checkpoint(self, ckpt_file):
        if ckpt_file is None:
            return None
        if isinstance(ckpt_file, str):
            with open(ckpt_file) as f:
                data = json.load(f)
        else:
            assert isinstance(ckpt_file, dict)
            data = ckpt_file
        return data

    def _copy(self, dst, src, prefix, allocate_tensor=False):
        if src is None:
            return src
        assert not dst.data.is_meta
        if allocate_tensor:
            dst = torch.empty_like(dst)
        src_shape = src.shape
        dst_shape = dst.shape
        for i in range(len(src_shape)):
            assert src_shape[i] >= dst_shape[i], "src shape should be larger or equal than dst shape in dim {}".format(i)
        assert len(src_shape) <= 2 and len(dst_shape) <= 2, "copy in weightload only support the two dim tensor, {} is not follow this rule".format(prefix)
        if len(src_shape) == 2:
            if src_shape == dst_shape:
                dst.data.copy_(src)
            elif src_shape[0] != dst_shape[0]:
                dst.data.copy_(src[self.gpu_idex * dst_shape[0]:(self.gpu_idex + 1) * dst_shape[0], :])
            else:
                dst.data.copy_(src[:, self.gpu_idex * dst_shape[1]:(self.gpu_idex + 1) * dst_shape[1]])
        else:
            if src_shape == dst_shape:
                dst.data.copy_(src)
            else:
                dst.data.copy_(src[self.gpu_idex * dst_shape[0]:(self.gpu_idex + 1) * dst_shape[0]])

        return dst

    def _qkv_copy(self,
                  dst,
                  src,
                  num_splits,
                  name,
                  allocate_tensor):
        if src is None:
            return src
        src_shape = src.shape
        dst_shape = dst.shape


        if allocate_tensor:
            dst = torch.empty_like(dst)
        src_split = torch.split(src.data, src.shape[1] // num_splits, dim=1)
        if (len(src_shape) == 2 and len(dst_shape) == 2):
            if src_shape[1] == dst_shape[1]:
                dst = dst.reshape(-1).data.copy_(src.data.reshape(-1)).reshape(src.shape)
                dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
                return dst
            assert src_shape[1] > dst_shape[1], "src shape should larger than dst shape when copy"
            qkv_size = dst_shape[1] // num_splits
            qkv_split = [torch.split(src_s, qkv_size, dim=1) for src_s in src_split]
            weight_split = [
                torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=1) for i in range(len(qkv_split[0]))
            ]
            dst = dst.reshape(-1).data.copy_(weight_split[self.gpu_idex].contiguous().reshape(-1)).reshape(
                weight_split[self.gpu_idex].shape)
        else:
            if src_shape[0] == dst_shape[0]:
                return torch.nn.parameter.Parameter(src)
            qkv_size = dst_shape[0] // num_splits
            qkv_split = [torch.split(src_s, qkv_size, dim=0) for src_s in src_split]
            bias_split = [torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=0) for i in range(len(qkv_split[0]))]
            dst.data.copy_(bias_split[self.gpu_idex].contiguous())

        dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
        return dst


    def _weight_load_scheme(self, module, name):
        if getattr(module, "replaced", False) == False:
            # Nothing happened in TensorSlicer, module should keeps no change.
            return
        tp_weight = getattr(module, "tp_weight", None)
        tp_bias = getattr(module, "tp_bias", None)
        model_weight = getattr(module, "weight", None)
        model_bias = getattr(module, "bias", None)
        assert tp_weight is not None, "Found no tp weight in the replaced module named {}".format(name)

        # load weight
        if model_weight is not None and tp_weight is not None:
            tp_shape = tp_weight.shape
            weight_shape = model_weight.shape
            for i in range(len(tp_shape)):
                if tp_shape[i] != weight_shape[i]:
                    assert weight_shape[i] // self.tp_size == tp_shape[i], "mp size is not consistant between WeightLoader and TensorSlicer, please check your config"
                    data = model_weight.data.split(tp_shape[i], dim=i)[self.gpu_idex].to(self.device)
                    param_weight = torch.nn.parameter.Parameter(data, requires_grad=False)
                    module.weight.data = param_weight
            setattr(module, "tp_weight", None)

        # load bias
        if model_bias is not None and tp_bias is not None:
            if model_bias.shape != tp_bias.shape:
                assert model_bias.shape // self.tp_size == tp_bias.shape, "mp size is not consistant between WeightLoader and TensorSlicer, please check your config"
                data = model_bias.data.split(tp_bias.shape, dim=0)[self.gpu_idex].to(self.device)
                para_bias = torch.nn.parameter.Parameter(data, requires_grad=False)
                module.bias.data = para_bias
            setattr(module, "tp_bias", None)
        return 

    def _load_weight_from_state_dict(self, module, state_dict, prefix):
        device = "xpu:{}".format(dist.get_rank())
        if hasattr(module, "weight"):
            meta_weight = getattr(module, "tp_weight", None)
            if meta_weight is None and module.weight.data.is_meta:
                meta_weight = module.weight.data
            if meta_weight is not None:
                data = torch.nn.parameter.Parameter(torch.empty_like(meta_weight, device="cpu"))
                module.weight = data
            sd_key = prefix + "weight"
            sd_weight = state_dict[sd_key]
            # if 'query_key_value' in prefix:
            #     self._qkv_copy(module.weight.data, sd_weight, 3, sd_key, "")
            # else:
            self._copy(module.weight.data, sd_weight, sd_key)
            module.weight.data = module.weight.data.to(device)
            setattr(module, "tp_weight", None)
        else:
            if hasattr(module, "norm") and hasattr(module.norm, 'weight'):
                if module.norm.weight.data.is_meta:
                    module.norm.weight = torch.nn.parameter.Parameter(torch.empty_like(module.norm.weight.data, device="cpu"))
                sd_key = prefix + "weight"
                sd_weight = state_dict[sd_key]
                self._copy(module.norm.weight.data, sd_weight, sd_key)
                module.norm.weight.data = module.norm.weight.data.to(device)

        if prefix + "bias" in state_dict.keys():
            if hasattr(module, "bias"):
                meta_bias = getattr(module, "tp_bias", None)
                if meta_bias is None and module.bias.data.is_meta:
                    meta_bias = module.bias.data
                if meta_bias is not None:
                    data = torch.nn.parameter.Parameter(torch.empty_like(meta_bias, device="cpu"))
                    module.bias = data
                sd_key = prefix + "bias"
                sd_bias = state_dict[sd_key]
                self._copy(module.bias.data, sd_bias, sd_key)
                module.bias.data = module.bias.data.to(device)
            else:
                if hasattr(module, 'norm') and hasattr(module.norm, 'bias'):
                    if module.norm.bias.data.is_meta:
                        module.norm.bias = torch.nn.parameter.Parameter(torch.empty_like(module.norm.bias.data, device="cpu"))
                    sd_key = prefix + "bias"
                    sd_bias = state_dict[sd_key]
                    self._copy(module.norm.bias, sd_bias, sd_key)
                    module.norm.bias = module.norm.bias.to(device)
        return

    def _load_buffer(self, module, state_dict, prefix):
        for name in module._buffers.keys():
            if module._buffers[name].data.is_meta:
                module._buffers[name] = torch.nn.parameter.Parameter(
                    data=torch.empty_like(module._buffers[name].data, device="cpu"),
                    requires_grad=module._buffers[name].data.requires_grad)
            if prefix + name in state_dict.keys():
                module._buffers[name].data.copy_(state_dict[prefix + name])
    
    def hidden_prefix_0_if_necessary(self, model, sd):
        model = str(model)
        key = re.search(r": (.*?)Model", model)
        if key is None:
            key = re.search(r": (.*?)Stack", model)
        if key is None:
            key = re.match(r"(.*?)Model", model)
        # if keys start with 'model.', don't skip level 0 prefix
        if sd != None:
            for item in sd.keys():
                if re.match("^model[.]", item):
                    return False
        if key is not None and key.group(1).lower() in ["bloom", "opt"]:
            return True
        return False

    def _load_weight_from_ckpt_recursive(self, model, sd, recursive_depth, prefix = ""):
        for name, child in model.named_children():
            check_name = prefix + name + "."
            check_name = "" if recursive_depth == 0 and self.hidden_prefix_0_if_necessary(model, sd) else check_name
            module_cls = [nn.Linear, nn.LayerNorm, nn.Embedding]
            if type(child) in module_cls and check_name != "" and any(check_name in item for item in sd.keys()):
                self._load_weight_from_state_dict(child, sd, check_name)
                if "word_embeddings." in check_name or "embed_tokens." in check_name or "wte." in check_name:
                    self.weight_embedding = child
            else:
                # self._load_attr_from_sliced_block(child)
                self._load_weight_from_ckpt_recursive(child, sd, recursive_depth + 1, check_name)
            if len(child._buffers) != 0 and any(check_name in item for item in sd.keys()):
                self._load_buffer(child, sd, check_name)

    def _load_attr_from_sliced_block(self, child):
        for param in [
                "n_heads", "inner_dim", "num_heads", "num_kv", "num_attention_heads", "num_attn_heads",
                "all_head_size", "embed_dim", "hidden_size", "num_key_value_heads"
        ]:
            tp_name = "tp_" + param
            if hasattr(child, tp_name):
                val = getattr(child, tp_name)
                setattr(child, param, val)

    def _load_weight_from_ckpt(self, model, ckpt):
        state_dict = torch.load(ckpt, map_location="cpu")
        self._load_weight_from_ckpt_recursive(model, state_dict, 0, "")
        embedding_weight = None
        for name, param in model.named_parameters():
            if "word_embeddings." in name or "embed_tokens." in name or "wte." in name:
                embedding_weight = param
        if embedding_weight is not None and hasattr(model, "lm_head") and hasattr(model.lm_head, "weight") and model.lm_head.weight.is_meta:
                model.lm_head.weight = embedding_weight


    def _load_weight_from_model(self, model):

        device = getattr(model, "device", None)
        if device is not None and device == 'meta':
            assert False, "Can not port data from a meta device model to our optimized model, please provide checkpoint path or load the model with real data type"
        for name, child in model.named_children():
            if type(child) in [nn.Linear, nn.LayerNorm, nn.Embedding]:
                self._weight_load_scheme(child, name)
            else:
                # self._load_attr_from_sliced_block(child)
                self._load_weight_from_model(child)


    def load_weight_if_necessary(self, model):
        if self.ckpt_dict is not None:
            ckpt = self.ckpt_dict["checkpoints"]
            progress = tqdm.tqdm(total=len(ckpt), desc=f"Loading {len(ckpt)} checkpoint shards")
            for elem in ckpt:
                self._load_weight_from_ckpt(model, elem)
                progress.update(1)
        elif hasattr(model, "slicing") and model.slicing is True:
            # if model already load the weight, port the actual data to slicing weight
            assert model.device != "meta", "This model is runing on tensor parallel mode on meta device but with no ckpt provided, please provide ckpt for running !"
            print("this model have been properly sliced in slicer, load weight from original model")
            self._load_weight_from_model(model)

        return