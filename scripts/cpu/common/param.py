class Param(object):
    def __init__(self):
        self._core_type = ''
        self._core_type_temp_ins = ''
        self._name = ''
        self._ipex_name = ''
        self._def_val = None
        self._is_vec = False
        self._vec_size = 0
        self._is_alias = False
        self._is_to_be_written = False
        self._is_tensor = False
        self._is_optional = False
        self._is_const = False
        self._is_ref = False
        self._is_pointer = False
        self._is_std_tuple = False
        self._is_std_vec = False
        self._sub_params = []

    @property
    def core_type(self):
        return self._core_type

    @core_type.setter
    def core_type(self, value):
        if value == 'Tensor':
            self._is_tensor = True
        if value == 'std::tuple':
            self._is_std_tuple = True
        if value== 'std::vector':
            self._is_std_vec = True
        self._core_type = value


    @property
    def core_type_temp_ins(self):
        return self._core_type_temp_ins

    @core_type_temp_ins.setter
    def core_type_temp_ins(self, value):
        self._core_type_temp_ins = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def ipex_name(self):
        return self._ipex_name

    @ipex_name.setter
    def ipex_name(self, value):
        self._ipex_name = value

    @property
    def def_val(self):
        return self._def_val

    @def_val.setter
    def def_val(self, value):
        self._def_val = value

    @property
    def is_vec(self):
        return self._is_vec

    @is_vec.setter
    def is_vec(self, value):
        self._is_vec = value

    @property
    def vec_size(self):
        return self._vec_size

    @vec_size.setter
    def vec_size(self, value):
        self._vec_size = value

    @property
    def is_alias(self):
        return self._is_alias

    @is_alias.setter
    def is_alias(self, value):
        self._is_alias = value

    @property
    def is_to_be_written(self):
        return self._is_to_be_written

    @is_to_be_written.setter
    def is_to_be_written(self, value):
        self._is_to_be_written = value

    @property
    def is_tensor(self):
        return self._is_tensor

    @is_tensor.setter
    def is_tensor(self, value):
        self._is_tensor = value

    @property
    def is_optional(self):
        return self._is_optional

    @is_optional.setter
    def is_optional(self, value):
        self._is_optional = value

    @property
    def is_const(self):
        return self._is_const

    @is_const.setter
    def is_const(self, value):
        self._is_const = value

    @property
    def is_ref(self):
        return self._is_ref

    @is_ref.setter
    def is_ref(self, value):
        self._is_ref = value

    @property
    def is_pointer(self):
        return self._is_pointer

    @is_pointer.setter
    def is_pointer(self, value):
        self._is_pointer = value

    @property
    def is_std_tuple(self):
        return self._is_std_tuple

    @is_std_tuple.setter
    def is_std_tuple(self, value):
        self._is_std_tuple = value

    @property
    def is_std_vec(self):
        return self._is_std_vec

    @is_std_vec.setter
    def is_std_vec(self, value):
        self._is_std_vec = value

    @property
    def sub_params(self):
        return self._sub_params

    @sub_params.setter
    def sub_params(self, value):
        self._sub_params = value
