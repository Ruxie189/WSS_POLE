import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import os.path as osp
from functools import partial
from net.vision_transformer import _cfg
import torch.nn as nn

# import mscoco.dataloader
from misc import torchutils, imutils
import net.resnet50_cam
import cv2
import voc12.dataloader

cudnn.enabled = True

def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    x = F.relu(x)

    b, c, h, w = x.size()

    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))

    return F.relu(x - epsilon) / (max_value + epsilon)

import cmapy
def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    if shape is not None:
        h, w, c = shape
        cam = cv2.resize(cam, (w, h))

    cam = cv2.applyColorMap(cam, cmapy.cmap('seismic'))
    return cam

def transpose(image):
    return image.transpose((1, 2, 0))
def denormalize(image, mean=None, std=None, dtype=np.uint8, tp=True):
    if tp:
        image = transpose(image)

    if mean is not None:
        image = (image * std) + mean

    if dtype == np.uint8:
        image *= 255.
        return image.astype(np.uint8)
    else:
        return image

def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def _work(process_id, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            print(img_name)
            label = pack['label'][0]
            size = pack['size']
            if iter % 500 == 0:
                print(f'[{iter} processed!]')

            path = os.path.join('/home/ruxie/scratch/ruxie/baseline/cam_mask',img_name+'.npy')
            print(path)
            cam_dict = np.load(path, allow_pickle = True).item()
            highres_cam = cam_dict['high_res']
            highres_cam = torch.tensor(highres_cam)


            cam = torch.sum(highres_cam, dim=0)
            cam = cam.unsqueeze(0).unsqueeze(0)

            cam = make_cam(cam).squeeze()
            cam = get_numpy_from_tensor(cam)

            image = np.array(pack['img'][0])[0]
            image = image[0]
            image = denormalize(image, imagenet_mean, imagenet_std)
            h, w, c = image.shape

            cam = (cam * 255).astype(np.uint8)
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
            cam = colormap(cam)
            #print(cam)

            image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
            cv2.imwrite(f'{args.work_space}/vis/{img_name}.png', image.astype(np.uint8))
            #cv2.imwrite(f'vis/{args.work_space}/{img_name}.png', image.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):

    if not os.path.exists(f'{args.work_space}/vis'):
        os.makedirs(f'{args.work_space}/vis')
    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                             scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()