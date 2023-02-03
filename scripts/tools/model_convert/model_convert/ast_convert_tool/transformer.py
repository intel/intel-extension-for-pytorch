import ast


class NodeTransformer(ast.NodeTransformer):
    """ main transformer for replacing string in ast nodes """

    make_restore: bool
    cuda_to_xpu_map: dict

    def __init__(self, make_restore=False):
        super().__init__()
        self.make_restore = make_restore
        self.cuda_to_xpu_map = {
            'cuda': 'xpu',
            'CUDA': 'XPU',
            'Cuda': 'XPU',
            'nccl': 'ccl',
            'has_cuda': 'True',
            b'cuda': b'xpu',
            b'CUDA': b'XPU',
            b'Cuda': b'XPU',
            b'nccl': b'ccl',
            b'has_cuda': b'True',
        }

    def _replace(self, obj):
        """ implementation of replacement and restorer """
        for key, val in self.cuda_to_xpu_map.items():
            (key, val) = (val, key) if self.make_restore else (key, val)
            if isinstance(obj, type(key)):
                obj = obj.replace(key, val)
        return obj

    def visit_Constant(self, node):
        """ replace `cuda` to `xpu` for ast.Constant """
        self.generic_visit(node)
        node.value = self._replace(node.value)
        return node

    def visit_Attribute(self, node):
        """ replace `cuda` to `xpu` for ast.Attribute """
        self.generic_visit(node)
        node.attr = self._replace(node.attr)
        return node

    def visit_FunctionDef(self, node):
        """ replace `cuda` to `xpu` for ast.FunctionDef """
        self.generic_visit(node)
        node.name = self._replace(node.name)
        return node

    def visit_ClassDef(self, node):
        """ replace `cuda` to `xpu` for ast.ClassDef """
        self.generic_visit(node)
        node.name = self._replace(node.name)
        return node

    def visit_Name(self, node):
        """ replace `cuda` to `xpu` for ast.Name """
        self.generic_visit(node)
        node.id = self._replace(node.id)
        return node

    def visit_arg(self, node):
        """ replace `cuda` to `xpu` for ast.arg """
        self.generic_visit(node)
        node.arg = self._replace(node.arg)
        return node

    def visit_alias(self, node):
        """ replace `cuda` to `xpu` for ast.arg """
        self.generic_visit(node)
        node.name = self._replace(node.name)
        node.asname = self._replace(node.asname)
        return node

    def visit_keyword(self, node):
        """ replace `cuda` to `xpu` for ast.keyword """
        self.generic_visit(node)
        node.arg = self._replace(node.arg)
        return node
# class NodeTransformer end.
