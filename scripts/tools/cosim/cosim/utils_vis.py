from collections import namedtuple
from distutils.version import LooseVersion
import graphviz
from graphviz import Digraph
import re
import itertools

import torch
from torch.autograd import Variable


Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    params = dict(params) if params is not None else None
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def tohex(val, nbits):
        return hex((val + (1 << nbits)) % (1 << nbits))

    output_nodes = (var.grad_fn,) if not isinstance(
        var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(
                    var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                grad_ptr = 'grad ptr: ' + str(tohex(u.grad.data_ptr(), 64)) if u.grad is not None else ''
                node_name = '%s\n %s %s' % (name, str(grad_ptr), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                dot.node(str(id(var)), str(type(var).__name__),
                         fillcolor='darkolivegreen1')
            else:
                dot.node(str(id(var)), var.name())
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    resize_graph(dot)

    return dot

# For traces


def replace(name, scope):
    return '/'.join([scope[name], name])


def parse(graph):
    scope = {}
    for n in graph.nodes():
        inputs = [i.uniqueName() for i in n.inputs()]
        for i in range(1, len(inputs)):
            scope[inputs[i]] = n.scopeName()

        uname = next(n.outputs()).uniqueName()
        assert n.scopeName() != '', '{} has empty scope name'.format(n)
        scope[uname] = n.scopeName()
    scope['0'] = 'input'

    nodes = []
    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}
        attrs = str(attrs).replace("'", ' ')
        inputs = [replace(i.uniqueName(), scope) for i in n.inputs()]
        uname = next(n.outputs()).uniqueName()
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': n.kind(),
                             'inputs': inputs,
                             'attr': attrs}))

    for n in graph.inputs():
        uname = n.uniqueName()
        if uname not in scope.keys():
            scope[uname] = 'unused'
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': 'Parameter',
                             'inputs': [],
                             'attr': str(n.type())}))

    return nodes


def make_dot_from_trace(trace):
    """ Produces graphs of torch.jit.trace outputs
    Example:
    >>> trace, = torch.jit.trace(model, args=(x,))
    >>> dot = make_dot_from_trace(trace)
    """
    # from tensorboardX
    if LooseVersion(torch.__version__) >= LooseVersion("0.4.1"):
        torch.onnx._optimize_trace(
            trace, torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    elif LooseVersion(torch.__version__) >= LooseVersion("0.4"):
        torch.onnx._optimize_trace(trace, False)
    else:
        torch.onnx._optimize_trace(trace)
    graph = trace.graph()
    list_of_nodes = parse(graph)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for node in list_of_nodes:
        dot.node(node.name, label=node.name.replace('/', '\n'))
        if node.inputs:
            for inp in node.inputs:
                dot.edge(inp, node.name)

    resize_graph(dot)

    return dot


# by Xu, Pinzhen
def draw(graph, show_constant=False, show_attr=False):

    index = itertools.count()
    graph_id = itertools.count()
    id_to_op_id = {}
    subgraphs = []

    def _stringify_values(values):
        return ['%' + v.debugName() for v in values]

    def _draw_subgraph(graph, g, graph_name=''):
        nonlocal index, graph_id, id_to_op_id, subgraphs
        name_to_id = {}

        if graph_name:
            g.attr(label=graph_name)

        # add graph inputs
        for input_name in _stringify_values(graph.inputs()):
            input_id = str(next(index))
            name_to_id[input_name] = input_id

            op_id = str(next(index))
            id_to_op_id[input_id] = op_id
            g.node(op_id, input_name)

        for node in graph.nodes():
            outputs_name = _stringify_values(node.outputs())
            inputs_name = _stringify_values(node.inputs())
            operation_name = node.kind()

            if not show_attr and operation_name == 'prim::GetAttr':
                continue
            if not show_constant and operation_name == 'prim::Constant':
                continue

            attrs = re.findall(r'=.+?(\[.+\])', str(node))
            operation_disp_name = operation_name + (attrs[0] if attrs else '')

            # find subgraph
            if node.hasAttribute('Subgraph'):
                subgraph = node.g('Subgraph')
                subgraph_name = 'subgraph_%d' % next(graph_id)
                subgraphs.append([subgraph, subgraph_name])
                operation_disp_name = operation_disp_name.replace(
                    'Subgraph=<Graph>', subgraph_name)

            # add operations
            op_id = str(next(index))
            g.node(op_id, operation_disp_name)

            # add outputs
            for output_name in outputs_name:
                output_id = str(next(index))
                name_to_id[output_name] = output_id
                id_to_op_id[output_id] = op_id

            # link inputs
            for input_name in inputs_name:
                if input_name in name_to_id and \
                        name_to_id[input_name] in id_to_op_id:
                    upstream_op_id = id_to_op_id[name_to_id[input_name]]
                    g.edge(upstream_op_id, op_id, label=input_name)

    dot = graphviz.Digraph()
    dot.format = 'svg'

    _draw_subgraph(graph, dot)
    while len(subgraphs):
        subgraph, subgraph_name = subgraphs.pop(0)
        with dot.subgraph(name='cluster_' + subgraph_name) as subdot:
            _draw_subgraph(subgraph, subdot, subgraph_name)

    return dot
