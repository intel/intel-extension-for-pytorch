import abc

class SigParser(metaclass=abc.ABCMeta):
    def __init__(self, _sig_str, _SIG_PARSER):
        self._sig_str = _sig_str
        self._sig_tree = _SIG_PARSER.parse(_sig_str)
        self._contain_alias_tensor = False
        self._contain_output_tensor = False
        self._def_name = ''
        self._is_tensor_member_func = False

        self._input_params = []
        self._ret_params = []

        self.get_all_input_params()
        self.get_all_return_params()

        for param in self._input_params:
            if param.is_alias:
                self._contain_alias_tensor = True
                break

        for param in self._input_params:
            if param.is_to_be_written:
                self._contain_output_tensor = True
                break

    @property
    def sig_str(self):
        return self._sig_str

    @property
    def sig_tree(self):
        return self._sig_tree

    @property
    def def_name(self):
        return self._def_name

    @def_name.setter
    def def_name(self, value):
        self._def_name = value

    @property
    def is_tensor_member_func(self):
        return self._is_tensor_member_func

    @is_tensor_member_func.setter
    def is_tensor_member_func(self, value):
        self._is_tensor_member_func = value

    @property
    def input_params(self):
        return self._input_params

    @property
    def ret_params(self):
        return self._ret_params

    @property
    def contain_alias_tensor(self):
        return len(self.get_alias_tensors()) > 0

    @property
    def contain_output_tensor(self):
        return len(self.get_output_tensors()) > 0

    def get_alias_tensors(self):
        alias_tensors = []
        for param in self._input_params:
            if param.is_alias:
                assert param.core_type == 'Tensor'
                alias_tensors.append(param)
        return alias_tensors

    def get_output_tensors(self):
        output_tensors = []
        for param in self._input_params:
            if param.is_to_be_written:
                assert param.core_type == 'Tensor'
                output_tensors.append(param)
        return output_tensors

    @abc.abstractmethod
    def get_all_input_params(self):
        # Child class should override this method
        raise AttributeError('Child class should override this method')

    @abc.abstractmethod
    def get_all_return_params(self):
        # Child class should override this method
        raise AttributeError('Child class should override this method')
