import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import intel_extension_for_pytorch as ipex

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, args):

    # switch to evaluate mode
    model.eval()

    def eval_func(model):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        number_iter = len(val_loader)

        progress = ProgressMeter(
            number_iter,
            [batch_time, losses, top1, top5],
            prefix='Test: ')
        print('Evaluating RESNET: total Steps: {}'.format(number_iter))
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.contiguous(memory_format=torch.channels_last)
                output = model(images)
                loss = criterion(output, target)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                if i % args.print_freq == 0:
                    progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg.item()

    print(".........runing calibration step.........")
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    qconfig = QConfig(
            activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
            weight= PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    x = torch.randn(1, 3, 224, 224)
    prepared_model = ipex.quantization.prepare(model, qconfig, x, inplace=True)
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.contiguous(memory_format=torch.channels_last)
            prepared_model(images)
            if i == 4:
                break
    print(".........calibration step done.........")

    print(".........runing autotuning step.........")
    tuned_model = ipex.quantization.autotune(prepared_model, val_loader, eval_func, sampling_sizes=[300])
    print(".........autotuning step done.........")

    print(".........runing int8 inference.........")
    converted_model = ipex.quantization.convert(tuned_model)
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.contiguous(memory_format=torch.channels_last)
            traced_model = torch.jit.trace(converted_model, images)
            traced_model = torch.jit.freeze(traced_model)
            break

    eval_func(traced_model)

    return

def main(args):
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)

    assert args.data != None, "please set dataset path if you want to using real data"
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    criterion = torch.nn.CrossEntropyLoss()

    val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
      ])),
      batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)

    validate(val_loader, model, criterion, args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=56, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    main(parser.parse_args())
