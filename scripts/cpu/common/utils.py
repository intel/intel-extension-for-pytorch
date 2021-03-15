import re

_TYPE_NSMAP = {
    'Tensor': 'at::Tensor', # Cover TensorList, TensorOptions and Tensor
    'Scalar': 'at::Scalar', # Cover ScalarType and Scalar
    'ScalarType': 'at::ScalarType', # Cover ScalarType and Scalar
    'Storage': 'at::Storage',
    'TensorList': 'at::TensorList',
    'TensorOptions': 'c10::TensorOptions',
    'IntList': 'at::IntList',
    'List': 'c10::List',
    'Stream': 'c10::Stream',
    'IntArrayRef': 'at::IntArrayRef',
    'ArrayRef': 'c10::ArrayRef',
    'Layout': 'c10::Layout',
    'Generator': 'at::Generator',
    'SparseTensorRef': 'at::SparseTensorRef',
    'Device': 'c10::Device',
    'optional': 'c10::optional',
    'MemoryFormat': 'at::MemoryFormat',
    'QScheme': 'at::QScheme',
    'ConstQuantizerPtr': 'at::ConstQuantizerPtr',
    'Dimname': 'at::Dimname',  # Cover DimnameList and Dimname
    'DimnameList': 'at::DimnameList',  # Cover DimnameList and Dimname
}


def add_ns(pt_string):
    splited_str = re.split(r'([^a-zA-Z0-9_])', pt_string)
    item_arr = []
    for item in splited_str:
        ns_item = _TYPE_NSMAP[item] if item in _TYPE_NSMAP.keys() else item
        item_arr.append(ns_item)
    return ''.join(item_arr)

def query_tensor_options(input_params):
    # c10::optional<at::ScalarType> dtype, c10::optional<c10::Layout> layout, c10::optional<c10::Device> device, c10::optional<bool> pin_memory
    start_idx = -1
    end_idx = -1
    for param in input_params:
        if param.is_optional:
            if param.name == "dtype" and param.core_type == "ScalarType":
                start_idx = input_params.index(param)
            elif param.name == "layout" and param.core_type == "Layout":
                pass
            elif param.name == "device" and param.core_type == "Device":
                pass
            elif param.name == "pin_memory" and param.core_type == "bool":
                end_idx = input_params.index(param)
                if start_idx >= 0:
                    return start_idx, end_idx
            else:
                start_idx = -1
        else:
            start_idx = -1
    return start_idx, end_idx

def is_out_func(fname):
    return fname.endswith("_out") or fname.endswith("_outf")

def reorder_params_idx(to_be_reordered_params, ref_params):
    new_idxs = {}
    assert len(to_be_reordered_params) == len(ref_params)
    for param in to_be_reordered_params:
        assert param in ref_params
        new_idxs[ref_params.index(param)] = to_be_reordered_params.index(param)
    return new_idxs

if __name__ == '__main__':
    sigs = [
    "aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
    "aten::abs_(Tensor(a!) self) -> Tensor(a!)",
    "aten::angle(Tensor self) -> Tensor",
    "aten::acos_(Tensor(a!) self) -> Tensor(a!)",
    "aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)",
    "aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)",
    "aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
    "aten::_test_optional_filled_intlist(Tensor values, int[2]? addends) -> Tensor",
    "Tensor upsample_nearest1d_backward(const Tensor & grad_output, c10::optional<IntArrayRef> output_size, IntArrayRef input_size, c10::optional<ArrayRef<double>> scale_factors);"
    ]

    for sig in sigs:
        print(add_ns(sig))
