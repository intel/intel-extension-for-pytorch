import torch
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

vocab_size = model.config.vocab_size
batch_size = 128
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)
######################################################  # noqa F401

with torch.no_grad():
    model(data)

print("Execution finished")
