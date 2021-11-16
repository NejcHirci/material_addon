import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    alb_path = os.path.join(args.dir, 'albedo.png')
    rough_path = os.path.join(args.dir, 'rough.png')
    normal_path = os.path.join(args.dir, 'normal.png')
    spec_path = os.path.join(args.dir, 'specular.png')

    albedo = transforms.ToTensor()(Image.open(alb_path).convert('RGB'))
    rough = transforms.ToTensor()(Image.open(rough_path).convert('RGB'))
    normal = transforms.ToTensor()(Image.open(normal_path).convert('RGB'))
    specular = transforms.ToTensor()(Image.open(spec_path).convert('RGB'))

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    alb_pred = batched_predict(model, ((albedo - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    alb_pred = (alb_pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(alb_pred).save(os.path.join(args.dir, 'albedo.png'))

    rough_pred = batched_predict(model, ((rough - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    rough_pred = (rough_pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(rough_pred).save(os.path.join(args.dir, 'rough.png'))

    normal_pred = batched_predict(model, ((normal - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    normal_pred = (normal_pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(normal_pred).save(os.path.join(args.dir, 'normal.png'))

    spec_pred = batched_predict(model, ((specular - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    spec_pred = (spec_pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(spec_pred).save(os.path.join(args.dir, 'specular.png'))
