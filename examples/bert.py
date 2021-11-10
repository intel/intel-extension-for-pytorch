import torch
from transformers import BertModel

def inference(model, data, torchscript):
  with torch.no_grad():
    if torchscript:
      model = torch.jit.trace(model, (data,), check_trace=False, strict=False)
      model = torch.jit.freeze(model)

    for _ in range(100):
      model(data)

    import time
    start = time.time()
    for _ in range(100):
      model(data)
    end = time.time()
    print('Inference took {:.2f} ms in average'.format((end-start)/100*1000))

def main(args):
  model = BertModel.from_pretrained(args.model_name)
  model.eval()

  vocab_size = model.config.vocab_size
  batch_size = 1
  seq_length = 512
  data = torch.randint(vocab_size, size=[batch_size, seq_length])

  import intel_extension_for_pytorch as ipex
  if args.dtype == 'float32':
    model = ipex.optimize(model, dtype=torch.float32, level="O1")
    inference(model, data, args.torchscript)
  if args.dtype == 'bfloat16':
    model = ipex.optimize(model, dtype=torch.bfloat16, level="O1")
    with torch.cpu.amp.autocast():
      inference(model, data, args.torchscript)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", default="bert-base-multilingual-cased")
  parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'int8'])
  parser.add_argument("--torchscript", default=False, action="store_true")

  main(parser.parse_args())
