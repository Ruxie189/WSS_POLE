import cv2

import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

import importlib

import voc12.dataloader
from misc import pyutils, torchutils, calutils
from torch import autograd
import os
#import nvidia_smi

calibration_techniques = ["MLSML", "ce", "ce_mixup", "focal", "focal_mixup", "mdca"]
mdca_beta = 1.0

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss': loss.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))

    return


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()

    #nvidia_smi.nvmlInit()
    #handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

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

    param_groups = model.trainable_parameters()
    print(len(param_groups))
    if args.cam_network == "net.resnet50_cam": optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
    elif args.cam_network == "net.mctformer_cam": optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
    elif args.cam_network == "net.resnet50_cam_adapt":optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[2], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    ce = nn.CrossEntropyLoss()
    focal = calutils.FocalLoss(gamma = 3.0)
    mdca = calutils.MDCA()
    cal_setup = args.calibration
    assert cal_setup in calibration_techniques, cal_setup+" is not a valid/available technique."

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda()

            if cal_setup in ["ce_mixup", "focal_mixup"]:
                img, label = calutils.mixup(img, label.argmax(dim=1), device="cuda", alpha=0.1, n_classes=len(voc12.dataloader.CAT_LIST))



            x = model(img)

            optimizer.zero_grad()

            if cal_setup in ["ce", "ce_mixup"]:
                loss = calutils.cross_entropy_loss(x, label)
            elif cal_setup in ["focal", "focal_mixup"]:
                loss = focal(x, label.argmax(dim=1))
            elif cal_setup == "mdca":
                loss = F.multilabel_soft_margin_loss(x, label) + mdca_beta * mdca(x, label.argmax(dim=1))
            else:
                loss = F.multilabel_soft_margin_loss(x, label)

            loss.backward()
            avg_meter.add({'loss': loss.item()})


            optimizer.step()
            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        
        validate(model, val_data_loader)
        timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name)
    torch.cuda.empty_cache()
