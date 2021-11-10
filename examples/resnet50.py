import torch
import torchvision.models as models

def inference(model, data, torchscript):
  with torch.no_grad():
    if torchscript:
      model = torch.jit.trace(model, data)
      model = torch.jit.freeze(model)

    for i in range(100):
      model(data)

    import time
    start = time.time()
    for i in range(100):
      output = model(data)
    end = time.time()
    print('Inference took {:.2f} ms in average'.format((end-start)/100*1000))

def main(args):
  model = models.resnet50(pretrained=False)
  model.eval()

  data = torch.rand(1, 3, 224, 224)

  import intel_extension_for_pytorch as ipex
  model = model.to(memory_format=torch.channels_last)
  data = data.to(memory_format=torch.channels_last)
  if args.dtype == 'float32':
    model = ipex.optimize(model, dtype=torch.float32, level='O1')
    inference(model, data, args.torchscript)
  if args.dtype == 'bfloat16':
    model = ipex.optimize(model, dtype=torch.bfloat16, level='O1')
    with torch.cpu.amp.autocast():
      inference(model, data, args.torchscript)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'int8'])
  parser.add_argument("--torchscript", default=False, action="store_true")

  main(parser.parse_args())
