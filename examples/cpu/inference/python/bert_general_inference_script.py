import torch
from transformers import BertModel

def inference(model, data):
    with torch.no_grad():
        # warm up
        for _ in range(10):
            model(data)

        # measure
        import time
        start = time.time()
        for _ in range(10):
            model(data)
        end = time.time()
        print('Inference took {:.2f} ms in average'.format((end - start) / 10 * 1000))

def main(args):
    model = BertModel.from_pretrained(args.model_name)
    model.eval()

    vocab_size = model.config.vocab_size
    batch_size = 128
    seq_length = 512
    data = torch.randint(vocab_size, size=[batch_size, seq_length])

    import intel_extension_for_pytorch as ipex

    if args.dtype == 'float32':
        model = ipex.optimize(model, dtype=torch.float32)
    elif args.dtype == 'bfloat16':
        model = ipex.optimize(model, dtype=torch.bfloat16)

    with torch.cpu.amp.autocast(enabled=args.dtype == 'bfloat16'):
        with torch.no_grad():
            model = torch.jit.trace(model, data, check_trace=False, strict=False)
            model = torch.jit.freeze(model)

        inference(model, data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-multilingual-cased")
    parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])

    main(parser.parse_args())

print("Execution finished")
