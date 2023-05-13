# reference: https://github.com/intel/neural-compressor/blob/\
#            15477100cef756e430c8ef8ef79729f0c80c8ce6/neural_compressor/conf/dotdict.py
class DotDict(dict):
    """access yaml using attributes instead of using the dictionary notation.
    Args:
        value (dict): The dict object to access.
    """

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError("expected dict")

    def __getitem__(self, key):
        value = self.get(key, None)
        return value

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            value = DotDict(value[0])
        if (
            isinstance(value, list)
            and len(value) > 1
            and all(isinstance(v, dict) for v in value)
        ):
            value = DotDict({k: v for d in value for k, v in d.items()})
        super(DotDict, self).__setitem__(key, value)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    __setattr__, __getattr__ = __setitem__, __getitem__
