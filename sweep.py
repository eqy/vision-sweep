import argparse
from progress.bar import Bar
import csv
import time

import torch
import torchvision

WARMUP_ITERS = 2
RUN_ITERS = 100

def generate_resolutions(base_factors=[56, 32]):
    resolutions = list()
    for base_factor in base_factors:
        for i in range(2, 9):
            x = base_factor*i
            for j in range(2, 9):
                y = base_factor*j
                resolutions.append((x,y))
    return resolutions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=str)
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--sku', type=str)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    if args.native:
        print("NOT USING CUDNN")
        torch.backends.cudnn.enabled = False

    if args.benchmark:
        assert not args.native
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False

    models_dict = torchvision.models.__dict__
    models = ['resnet18',
              'alexnet',
              'vgg16',
              'squeezenet1_0',
              'densenet161',
              #'inception_v3',
              'googlenet',
              'shufflenet_v2_x1_0', 
              'mobilenet_v2',
              'mobilenet_v3_large',
              'mobilenet_v3_small',
              'resnext50_32x4d',
              'wide_resnet50_2',
              'mnasnet1_0']
    output = {'model': [], 'resolution': [], 'iter_time': []}

    assert args.sku in ['3080', '3090', 'A100', 'V100', 'A6000']
    bar = Bar('models', max=len(models))

    for idx, model in enumerate(models):
        batch_size = 64
        if args.sku not in ['A100', 'A6000']:
            batch_size = 32 
        if args.sku in ['3080']:
            batch_size = 12
        if args.sku in ['V100']:
            batch_size = 48

        resolutions = generate_resolutions()
        m = models_dict[model]().cuda()
        if 'densenet' in model or 'wide_resnet' in model:
            batch_size //= 4
        elif 'resnext50_32x4d' in model:
            batch_size //= 4
        for resolution in resolutions:
            data = torch.rand(batch_size, 3, resolution[0], resolution[1], device='cuda')
            for i in range(WARMUP_ITERS):
                o = m(data)
            torch.cuda.synchronize()
            t1 = time.time()
            for i in range(RUN_ITERS):
                o = m(data)
            torch.cuda.synchronize()
            t2 = time.time()
            iter_time = (t2-t1)/RUN_ITERS
            output['model'].append(model)
            output['resolution'].append(resolution)
            output['iter_time'].append(iter_time)
        bar.next()
        if args.dry_run and idx >= 2:
            break
    bar.finish()
    if args.output is not None:
        assert len(output['model']) == len(output['resolution'])
        assert len(output['iter_time']) == len(output['model'])
        with open(args.output, 'w') as f:
            csvwriter = csv.writer(f)
            for idx, model in enumerate(output['model']):
                row = (model, output['resolution'][idx], output['iter_time'][idx])
                csvwriter.writerow(row)

if __name__ == '__main__':
    main()
