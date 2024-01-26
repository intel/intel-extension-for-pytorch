import torch
import torchvision.models as models

def inference(model, data):
    with torch.no_grad():
        # warm up
        for _ in range(100):
            model(data)

        # measure
        import time
        start = time.time()
        for _ in range(100):
            output = model(data)
        end = time.time()
        print('Inference took {:.2f} ms in average'.format((end - start) / 100 * 1000))

def main(args):
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model.eval()

    data = torch.rand(128, 3, 224, 224)

    import intel_extension_for_pytorch as ipex

    model = model.to(memory_format=torch.channels_last)
    data = data.to(memory_format=torch.channels_last)

    if args.dtype == 'float32':
        model = ipex.optimize(model, dtype=torch.float32)
    elif args.dtype == 'bfloat16':
        model = ipex.optimize(model, dtype=torch.bfloat16)
    else:  # int8
        from intel_extension_for_pytorch.quantization import prepare, convert

        qconfig = ipex.quantization.default_static_qconfig_mapping
        model = prepare(model, qconfig, example_inputs=data, inplace=False)

        # calibration
        n_iter = 100
        with torch.no_grad():
            for i in range(n_iter):
                model(data)

        model = convert(model)

    with torch.cpu.amp.autocast(enabled=args.dtype == 'bfloat16'):
        with torch.no_grad():
            model = torch.jit.trace(model, data)
            model = torch.jit.freeze(model)

        inference(model, data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'int8'])

    main(parser.parse_args())

print("Execution finished")
