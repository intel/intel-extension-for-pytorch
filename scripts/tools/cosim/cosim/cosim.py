import os
import time
from collections import OrderedDict, Iterable
from functools import partial

from graphviz import Digraph

import torch
from torch import nn

from .utils_vis import resize_graph


def mapping_to(obj, device, dtype):
    func1D = partial(mapping_to, device=device, dtype=dtype)
    func2D = lambda kv: (kv[0], func1D(kv[1]))
    if isinstance(obj, torch.Tensor):
        ret = obj.detach().clone().to(device=device, dtype=dtype)
        ret.requires_grad = obj.requires_grad
        return ret
    elif isinstance(obj, tuple) and len(obj) > 0:
        return tuple(map(func1D, obj))
    elif isinstance(obj, list) and len(obj) > 0:
        return list(map(func2D, obj))
    elif isinstance(obj, dict) and len(obj) > 0:
        return dict(map(func2D, obj.items()))
    elif isinstance(obj, OrderedDict) and len(obj) > 0:
        return OrderedDict(map(func2D, obj.items()))
    return obj

def clone_module(module):
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                cloned = param.clone()
                clone._parameters[param_key] = cloned

    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None:
                buff = module._buffers[buffer_key]
                cloned = buff.clone()
                clone._buffers[buffer_key] = cloned

    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key]
            )

    # if hasattr(clone, 'flatten_parameters'):
    #     clone = clone._apply(lambda x: x)
    return clone


CALLED_NODES = []
NODE_ID_MAP = {}
UUID = 0

class CosimModule(nn.Module):
    def __init__(self, m, tDev='cpu', tDtype=torch.float32):
        super().__init__()
        try:
            torch.tensor(1, device=tDev, dtype=tDtype)
        except Exception:
            errmsg = f"Can't create test tensor with device {tDev} and dtype {tDtype}"
            assert False, errmsg
        finally:
            self.target_device = tDev
            self.target_dtype  = tDtype
            # for now we aspect that user always want to enable cosim test
            self.enabled = True
            # for now we disable inf, nan check for temporary
            self.check_inf = False
            self.check_nan = False
            
        self.modules_path_list = []
        for name, mod in m.named_modules():
            self.modules_path_list.append(name)
        self.top_module = self.wrap_modules(m)
        self.idx = 0
        self.set_names(self.top_module)

    def _clear_cosim(self):
        global CALLED_NODES, NODE_ID_MAP, UUID
        for node in CALLED_NODES:
            node._clear()
        CALLED_NODES = []
        NODE_ID_MAP = {}
        UUID = 0



    def forward(self, *inputs, **kwargs):
        return self.top_module(*inputs, **kwargs)

    def wrap_modules(self, cur_m):
        if cur_m._modules:
            cur_m._modules = OrderedDict(
                map(lambda kv: (
                    kv[0],
                    self.wrap_modules(kv[1])),
                    cur_m._modules.items()))
            return CosimWrapper(cur_m, self.target_device, self.target_dtype, False)
        else:
            return CosimWrapper(cur_m, self.target_device, self.target_dtype, True)
    
    def set_names(self, cur_module):
        assert isinstance(cur_module, CosimWrapper), \
            f"Current module not be wrapped. type(cur_module) is {type(cur_module)}"
        cur_module._set_name(self.modules_path_list[self.idx])
        self.idx += 1
        for module in cur_module.cosim_module._modules.values():
            self.set_names(module)


    def plot_result(self, file="cosim_outputs/"):
        def _cal_max_diff(origin_outputs, cosim_outputs):
            # for now, atol & rtol are set to float32 default tolerance
            # will support customized tol in the future
            atol = 1e-5
            rtol = 1e-3

            def get_tensor_diff(a, b):
                a_cpu = a.cpu()
                b_cpu = b.cpu()
                absdiff = torch.abs(a_cpu - b_cpu)
                tolerance = atol + rtol * torch.abs(b_cpu)
                diffmask = absdiff > tolerance
                errors = absdiff[diffmask]
                return max(errors) if errors.numel() > 0 else 0.
            
            def get_scalar_diff(a, b):
                absdiff = abs(a - b)
                tolerance = atol + rtol * abs(b)
                if absdiff > tolerance:
                    return absdiff
                else:
                    return 0.


            diffs = []
            assert type(origin_outputs) == type(cosim_outputs), \
                f"origin_output's type({type(origin_outputs)}) is not equal" \
                f" to cosim output's type({type(cosim_outputs)})." \
                f" Please report to cosim tool's developer(xunsong.huang@intel.com)"
            if origin_outputs is None:
                diffs.append(None)
            elif isinstance(origin_outputs, torch.Tensor) \
                    and isinstance(cosim_outputs, torch.Tensor):
                diffs.append(get_tensor_diff(origin_outputs, cosim_outputs))
            else:
                # experimental, not tested yet
                for oo, co in zip(origin_outputs, cosim_outputs):
                    assert type(oo) == type(co), \
                        f"origin_output's type({type(oo)}) is not equal" \
                        f" to cosim output's type({type(co)})." \
                        f" Please report to cosim tool's developer" \
                        f"(xunsong.huang@intel.com)"
                    if oo is None:
                        diffs.append(None)
                    elif isinstance(oo, torch.Tensor):
                        diffs.append(get_tensor_diff(oo, co))
                    elif isinstance(oo, (tuple, list)):
                        for item1, item2 in zip(oo, co):
                            diffs.append(_cal_max_diff(item1, item2))
                    else:
                        diffs.append(get_scalar_diff(oo, co))
            return diffs

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot= Digraph(node_attr=node_attr, graph_attr=dict(size='12,12',
                                                           rankdir='RL',))

        def gen_unique_id(cur_path):
            global UUID
            UUID += 1
            return f"{str(id(cur_path))}+{cur_path}+{str(UUID)}"

        def add_node(cur_path, msg, flag):
            global NODE_ID_MAP
            node_id = gen_unique_id(cur_path)
            if '.' not in cur_path:
                print("got root path: ", cur_path)
                dot.node(node_id, cur_path, fillcolor='lightblue')
            else:
                parent = cur_path.rsplit('.', 1)[0]
                print("cur_path is: ", cur_path)
                print("parent is: ", parent)
                if parent not in NODE_ID_MAP:
                    add_node(parent, "", False)
                if msg == "":
                    msg = f"NODE: {cur_path}"
                    dot.node(node_id, msg, fillcolor='lightblue')
                else:
                    if flag:
                        dot.node(node_id, msg, fillcolor='pink')
                    else:
                        dot.node(node_id, msg, fillcolor='green')
                dot.edge(node_id, NODE_ID_MAP[parent])
            NODE_ID_MAP[cur_path] = node_id


        global CALLED_NODES

        for node in CALLED_NODES:
            if node.cosim_outputs_list:
                print(f"AT node {node.cosim_path}:")
                over_error = False
                msg = f"NODE: {node.cosim_path}\n"
                print("-> This node has cosim comparasion results.")
                origin_outputs = node.outputs_list[node.visited]
                cosim_outputs = node.cosim_outputs_list[node.visited]
                output_diffs = _cal_max_diff(origin_outputs, cosim_outputs)
                for i in range(len(output_diffs)):
                    print(f"\toutput #{i}'s max error: {output_diffs[i]}")
                    msg += f"output #{i}'s max error: {output_diffs[i]}\n"
                    if output_diffs[i] is not None and output_diffs[i] > 0.:
                        over_error = True
                print("-> This node has cosim grad comparasion results for inputs.")
                origin_grads = node.grads_list
                cosim_grads = node.cosim_grads_list
                grad_diffs = _cal_max_diff(origin_grads, cosim_grads)
                for i in range(len(grad_diffs)):
                    print(f"\tgrad of input #{i}'s max error: {grad_diffs[i]}")
                    msg += f"grad of input #{i}'s max error: {grad_diffs[i]}\n"
                    if grad_diffs[i] is not None and \
                            isinstance(grad_diffs[i], Iterable):
                        for g in grad_diffs[i]:
                            if g is not None and g > 0.:
                                over_error = True
                    elif grad_diffs[i] is not None and grad_diffs[i] > 0.:
                        over_error = True
                add_node(node.cosim_path, msg, over_error)
            node.visited += 1

        resize_graph(dot)
        file = os.path.join(file,
                            str(time.time_ns()), 
                            "cosim_result.gv")
        self._clear_cosim()
        return dot.render(file)


class CosimWrapper(nn.Module):
    def __init__(self, m, device, dtype, is_leaf):
        super().__init__()
        self.cosim_module = m
        self.cosim_path = ""
        self.inputs_list = []
        self.cosim_inputs_list = []
        self.kwargs_list = []
        self.cosim_kwargs_list = []
        self.outputs_list = []
        self.cosim_outputs_list = []
        self.grads_list = []
        self.cosim_grads_list = []
        self.should_do_cosim = is_leaf
        self.target_device = device
        self.target_dtype = dtype
        self.visited = 0

    def __getitem__(self, idx):
        return self.cosim_module[idx]

    def _set_name(self, name):
        self.cosim_path = name

    def _clear(self):
        self.inputs_list = []
        self.cosim_inputs_list = []
        self.kwargs_list = []
        self.cosim_kwargs_list = []
        self.outputs_list = []
        self.cosim_outputs_list = []
        self.grads_list = []
        self.cosim_grads_list = []
        self.visited = 0

    def register_bwd_hook(self, outputs, cosim_outputs=None):

        cur_output = None
        cur_cosim_output = None

        def _backward_hook(grad):
            if self.should_do_cosim:
                cosim_grad = grad.detach().clone().to(
                    self.target_device, self.target_dtype)
                if cur_cosim_output is not None:
                    self.cosim_grads_list.append(cur_cosim_output.grad_fn(cosim_grad))
                print(self.cosim_path, "'s cosim grad is computed.'")
            else:
                print(self.cosim_path, "'s grad is computed.")
            cur_grad = grad.detach().clone()
            self.grads_list.append(cur_output.grad_fn(cur_grad))
            return grad

        if isinstance(outputs, torch.Tensor) and \
                outputs.requires_grad is True:
            cur_output = outputs
            cur_output.retain_grad()
            cur_cosim_output = cosim_outputs
            if cur_cosim_output is not None:
                cur_cosim_output.retain_grad()
            cur_output.register_hook(_backward_hook)
        else:
            for o in outputs:
                if isinstance(o, torch.Tensor) and \
                        o.requires_grad is True:
                    cur_output = o
                    cur_output.retain_grad()
                    cur_cosim_output = cosim_outputs[outputs.index(o)] \
                        if cosim_outputs is not None else None
                    if cur_cosim_output is not None:
                        cur_cosim_output.retain_grad()
                    cur_output.register_hook(_backward_hook)

    def forward(self, *inputs, **kwargs):
        global CALLED_NODES

        CALLED_NODES.append(self)
        self.inputs_list.append(inputs)
        self.kwargs_list.append(kwargs)
        if self.should_do_cosim:
            t_dev = self.target_device
            t_dtype = self.target_dtype
            cosim_inputs = mapping_to(inputs, device=t_dev, dtype=t_dtype)
            self.cosim_inputs_list.append(cosim_inputs)
            cosim_kwargs = mapping_to(kwargs, device=t_dev, dtype=t_dtype)
            self.cosim_kwargs_list.append(cosim_kwargs)
            cosim_layer = CosimWrapper(
                clone_module(self.cosim_module), t_dev, t_dtype, False)
            cosim_layer.load_state_dict(self.state_dict())
            cosim_layer.to(t_dev, t_dtype)
            self.cosim_outputs_list.append(
                cosim_layer(*self.cosim_inputs_list[-1],
                            **self.cosim_kwargs_list[-1]))
        self.outputs_list.append(
            self.cosim_module(*self.inputs_list[-1],
                              **self.kwargs_list[-1]))
        if self.should_do_cosim:
            self.register_bwd_hook(self.outputs_list[-1], self.cosim_outputs_list[-1])
        else:
            self.register_bwd_hook(self.outputs_list[-1])
        return self.outputs_list[-1]

