import torch
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

vocab_size = model.config.vocab_size
batch_size = 1
seq_length = 512
data = torch.randint(vocab_size, size=[batch_size, seq_length])
torch.manual_seed(43)

#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex
model, optimizer = ipex.fast_bert(model, optimizer=optimizer, dtype=torch.bfloat16)
######################################################  # noqa F401

with torch.cpu.amp.autocast(dtype=torch.bfloat16):
  labels = torch.tensor(1)
  outputs = model(data, labels=labels)
  loss = outputs.loss
  loss.backward()
  optimizer.step()

print("Execution finished")
