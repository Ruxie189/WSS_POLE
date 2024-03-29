
import numpy as np
import os
import json
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from clip_tools.clip_utils import category_dict

def print_iou(iou, dname='voc'):
    iou_dict = {}
    for i in range(len(iou)-1):
        iou_dict[category_dict[dname][i]] = iou[i+1]
    print(iou_dict)

    return iou_dict

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    labels = []
    n_images = 0
    miou_save = {}
    
    #file_name = open('/home/ruxie/projects/def-josedolz/ruxie/CLIP-WSL/voc12/new_infer_clims.txt', 'r')
    #Lines = file_name.readlines()
    #ids = []
    #for line in Lines:
        #ids.append(line.replace('\n',''))
    #print(ids[0])
    for i, id in enumerate(dataset.ids):
        #print(type(id))
        n_images += 1
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
   
        preds.append(cls_labels.copy())
        labels.append(dataset.get_example_by_keys(i, (1,))[0])
    
        miou_save[id]={"pred":cls_labels.copy().tolist(),"label":dataset.get_example_by_keys(i, (1,))[0].tolist()}
        #break
    f = open(args.work_space + "/save_miou.json", 'w') 
    #f = open("/home/ruxie/scratch/ruxie/test_LOTMBTM_recheck_sen2/save_miou.json", 'w')  #insert a suitable path for saving the json
    f.write(json.dumps(miou_save))
    f.close()
    #print(len(preds))
    confusion = calc_semantic_segmentation_confusion(preds, labels)
    #print(confusion.shape)

    #new confusion
    '''
    exclusion_classes = [6,8,9,11,18]
    new_conf =  np.zeros([16,16])
    a,b = 0,0
    for i in range(21):
        if i in exclusion_classes: continue
    for j in range(21):
        if j in exclusion_classes: continue
        new_conf[a,b] = confusion[i,j]
        b += 1
    a += 1
    b = 0
    '''
    gtj = confusion.sum(axis=1)
    #gtj = new_conf.sum(axis=1)
    resj = confusion.sum(axis=0)
    #resj = new_conf.sum(axis=0)
    gtjresj = np.diag(confusion)
    #gtjresj = np.diag(new_conf)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    # print(iou)
    iou_dict = print_iou(iou, 'voc')

    print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)
    print('among_predfg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))

    hyper = [float(h) for h in args.hyper.split(',')]
    name = args.clims_weights_name + f'_{hyper[0]}_{hyper[1]}_{hyper[2]}_{hyper[3]}_ep({args.clims_num_epoches})_lr({args.clims_learning_rate}).pth'
    with open(args.work_space + '/eval_result.txt', 'a') as file:
        file.write(name + f' {args.clip} th: {args.cam_eval_thres}, mIoU: {np.nanmean(iou)} {iou_dict} \n')

    return np.nanmean(iou)
