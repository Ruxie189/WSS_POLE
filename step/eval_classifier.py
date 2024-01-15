import cv2

import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils, calutils
from torch import autograd
import os
from functools import partial
from net.vision_transformer import _cfg

import numpy as np
import sklearn.metrics as M
from tqdm import tqdm

def evaluate(model, data_loader):
    print('Evaluating ... ', flush=True, end='')

    model.eval()
    LABELS = []
    PREDICTIONS = []
    with torch.no_grad():
        for pack in tqdm(data_loader, total=len(data_loader)):
            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)
            x = model(img)
            LABELS += label.argmax(dim=1).detach().cpu().numpy().tolist()
            PREDICTIONS += F.softmax(x, dim=1).argmax(dim=1).cpu().numpy().tolist()
    acc = M.accuracy_score(LABELS, PREDICTIONS)
    return acc

def run(args):
    if args.cam_network == "net.resnet50_clims": 
        model = getattr(importlib.import_module(args.cam_network), 'Net')()
        model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    elif args.cam_network == "net.mctformer_clims": 
        model = getattr(importlib.import_module(args.cam_network), 'Net')(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model.default_cfg = _cfg()
        model.load_state_dict(torch.load(args.cam_weights_name)['model'], strict=True)
    elif args.cam_network == "net.conformer_cam":
        model = getattr(importlib.import_module(args.cam_network), 'Net')(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
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
    
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    model = torch.nn.DataParallel(model).cuda()
    train_acc = evaluate(model, train_data_loader)
    val_acc = evaluate(model, val_data_loader)
    print("Train set accuracy:", train_acc, ", Val set accuracy:", val_acc)
    torch.cuda.empty_cache()