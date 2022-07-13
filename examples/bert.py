import torch
from transformers import BertModel

def inference(model, data):
  with torch.no_grad():
    # warm up
    for _ in range(100):
      model(data)
    
    # measure
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
    model = ipex.optimize(model, dtype=torch.float32)
  elif args.dtype == 'bfloat16':
    model = ipex.optimize(model, dtype=torch.bfloat16)
  else: # int8
    from intel_extension_for_pytorch.quantization import prepare, convert
    
    if args.quantization == 'static':
        qconfig = ipex.quantization.default_static_qconfig
        model = prepare(model, qconfig, example_inputs=data, inplace=False)
        
        # calibration 
        n_iter = 100
        for i in range(n_iter):
            model(data)
            
        model = convert(model)
    else:
        qconfig = ipex.quantization.default_dynamic_qconfig
        model = prepare(model, qconfig, example_inputs=data)
        model = convert(model)
  
  if args.torchscript:
    with torch.no_grad():
        model = torch.jit.trace(model, data, check_trace=False, strict=False)
        model = torch.jit.freeze(model)
  
  with torch.cpu.amp.autocast(enabled=args.dtype=='bfloat16'):
      inference(model, data)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", default="bert-base-multilingual-cased")
  parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'int8'])
  parser.add_argument("--torchscript", default=False, action="store_true")
  parser.add_argument('--quantization', default='static', choices=['static', 'dynamic'])

  main(parser.parse_args())
