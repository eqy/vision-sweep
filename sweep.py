import argparse
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
    args = parser.parse_args()

    if args.native:
        print("NOT USING CUDNN")
        torch.backends.cudnn.enabled = False

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

    for idx, model in enumerate(models):
        resolutions = generate_resolutions()
        m = models_dict[model]().cuda()
        batch_size = 64
        if 'densenet' in model or 'wide_resnet' in model:
            batch_size = 16
        elif 'resnext50_32x4d' in model:
            batch_size = 16
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
            print(f'{model} {resolution} {iter_time}')
            output['model'].append(model)
            output['resolution'].append(resolution)
            output['iter_time'].append(iter_time)
        if args.dry_run and idx >= 2:
            break
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
