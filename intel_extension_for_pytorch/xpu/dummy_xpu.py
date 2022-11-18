import intel_extension_for_pytorch._C
from ._utils import _dummy_type


if not hasattr(intel_extension_for_pytorch._C, 'ShortStorageBase'):
    intel_extension_for_pytorch._C.__dict__['ShortStorageBase'] = _dummy_type('ShortStorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'CharStorageBase'):
    intel_extension_for_pytorch._C.__dict__['CharStorageBase'] = _dummy_type('CharStorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'IntStorageBase'):
    intel_extension_for_pytorch._C.__dict__['IntStorageBase'] = _dummy_type('IntStorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'LongStorageBase'):
    intel_extension_for_pytorch._C.__dict__['LongStorageBase'] = _dummy_type('LongStorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'BoolStorageBase'):
    intel_extension_for_pytorch._C.__dict__['BoolStorageBase'] = _dummy_type('BoolStorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'HalfStorageBase'):
    intel_extension_for_pytorch._C.__dict__['HalfStorageBase'] = _dummy_type('HalfStorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'DoubleStorageBase'):
    intel_extension_for_pytorch._C.__dict__['DoubleStorageBase'] = _dummy_type('DoubleStorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'FloatStorageBase'):
    intel_extension_for_pytorch._C.__dict__['FloatStorageBase'] = _dummy_type('FloatStorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'BFloat16StorageBase'):
    intel_extension_for_pytorch._C.__dict__['BFloat16StorageBase'] = _dummy_type('BFloat16StorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'QUInt8StorageBase'):
    intel_extension_for_pytorch._C.__dict__['QUInt8StorageBase'] = _dummy_type('QUInt8StorageBase')
if not hasattr(intel_extension_for_pytorch._C, 'QInt8StorageBase'):
    intel_extension_for_pytorch._C.__dict__['QInt8StorageBase'] = _dummy_type('QInt8StorageBase')
