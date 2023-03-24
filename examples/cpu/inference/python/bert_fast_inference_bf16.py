import torch
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])
torch.manual_seed(43)

#################### code changes ####################
import intel_extension_for_pytorch as ipex
model = ipex.fast_bert(model, dtype=torch.bfloat16)
######################################################

with torch.no_grad():
  model(data)
