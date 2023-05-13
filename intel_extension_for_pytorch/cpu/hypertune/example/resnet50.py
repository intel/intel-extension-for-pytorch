import torch
import torchvision.models as models


def inference(model, data):
    with torch.no_grad():
        # warm up
        for _ in range(100):
            model(data)

        # measure
        import time

        measure_iter = 100
        start = time.time()
        for _ in range(measure_iter):
            output = model(data)
        end = time.time()

        duration = (end - start) * 1000
        latency = duration / measure_iter
        throughput = measure_iter / duration

        print(
            "@hypertune {'name': 'latency (ms)'}"
        )  # Add print statement of the form @hypertune {'name': str, 'higher_is_better': bool, 'target_val': int or float}`
        print(
            latency
        )  # Print the objective(s) you want to optimize. Make sure this is just an int or float to be minimzied or maximized.


def main(args):
    model = models.resnet50(pretrained=False)
    model.eval()

    data = torch.rand(1, 3, 224, 224)

    import intel_extension_for_pytorch as ipex

    model = model.to(memory_format=torch.channels_last)
    data = data.to(memory_format=torch.channels_last)

    if args.dtype == "float32":
        model = ipex.optimize(model, dtype=torch.float32)
    elif args.dtype == "bfloat16":
        model = ipex.optimize(model, dtype=torch.bfloat16)
    else:  # int8
        from intel_extension_for_pytorch.quantization import prepare, convert

        qconfig = ipex.quantization.default_static_qconfig
        model = prepare(model, qconfig, example_inputs=data, inplace=False)

        # calibration
        n_iter = 100
        for i in range(n_iter):
            model(data)

        model = convert(model)

    with torch.cpu.amp.autocast(enabled=args.dtype == "bfloat16"):
        if args.torchscript:
            with torch.no_grad():
                model = torch.jit.trace(model, data)
                model = torch.jit.freeze(model)

        inference(model, data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "bfloat16", "int8"]
    )
    parser.add_argument("--torchscript", default=False, action="store_true")

    main(parser.parse_args())
