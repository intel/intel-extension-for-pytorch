import torch
#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
######################################################  # noqa F401

##### Example Model #####  # noqa F401
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

vocab_size = model.config.vocab_size
batch_size = 128
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])
#########################  # noqa F401

qconfig_mapping = ipex.quantization.default_dynamic_qconfig_mapping
# Alternatively, define your own qconfig:
# from torch.ao.quantization import PerChannelMinMaxObserver, PlaceholderObserver, QConfig, QConfigMapping
# qconfig = QConfig(
#        activation = PlaceholderObserver.with_args(dtype=torch.float, is_dynamic=True),
#        weight = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
# qconfig_mapping = QConfigMapping().set_global(qconfig)
prepared_model = prepare(model, qconfig_mapping, example_inputs=data)

converted_model = convert(prepared_model)
with torch.no_grad():
    traced_model = torch.jit.trace(converted_model, (data,), check_trace=False, strict=False)
    traced_model = torch.jit.freeze(traced_model)

traced_model.save("dynamic_quantized_model.pt")

print("Saved model to: dynamic_quantized_model.pt")
