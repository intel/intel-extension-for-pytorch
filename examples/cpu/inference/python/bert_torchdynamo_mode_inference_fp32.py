import torch
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])

# Experimental Feature
#################### code changes ####################
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)
model = torch.compile(model, backend="ipex")
######################################################

with torch.no_grad():
  model(data)
