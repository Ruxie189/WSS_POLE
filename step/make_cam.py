import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.backends import cudnn

import numpy as np
import importlib
import os
import os.path as osp
from functools import partial
from net.vision_transformer import _cfg

import voc12.dataloader
from misc import torchutils, imutils
import cv2
cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']] # b x 20 x w x h

            #print(len(outputs))
            #for i, o in enumerate(outputs): print(pack['img'][i].shape, o.shape)

            strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            
            np.save(os.path.join(args.cam_out_dir, img_name.replace('jpg','npy')),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    if args.cam_network == "net.resnet50_clims": 
        model = getattr(importlib.import_module(args.cam_network), 'CAM')()
        #model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
        model.load_state_dict(torch.load('cam-baseline-voc12/res50_cam.pth'), strict=True)
    elif args.cam_network == "net.mctformer_cam":
        model = getattr(importlib.import_module(args.cam_network), 'CAM')(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model.default_cfg = _cfg()
        model.load_state_dict(torch.load(args.cam_weights_name)['model'], strict=True)
    elif args.cam_network == "net.conformer_cam":
        model = getattr(importlib.import_module(args.cam_network), 'CAM')(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, num_classes=21)
        checkpoint = torch.load('cam-baseline-voc12/Conformer_small_patch16.pth', map_location='cpu')
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        else:
            checkpoint = checkpoint
        model_dict = model.state_dict()
        for k in ['trans_cls_head.weight', 'trans_cls_head.bias']:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
        for k in ['conv_cls_head.weight', 'conv_cls_head.bias']:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif args.cam_network == "net.resnet50_cam_adapt":
         model = getattr(importlib.import_module(args.cam_network), 'CAM')()
         model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
        #model.load_state_dict(torch.load('cam-baseline-voc12/res50_cam.pth'), strict=True)

    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list, voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()