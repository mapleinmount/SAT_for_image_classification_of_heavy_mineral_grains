import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
from util.DomainImageFolder import DomainImageFolder
from SAN import SAN

from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, cal_accuracy
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, cohen_kappa_score,confusion_matrix
import pandas as pd
import ttach as tta
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='Heavy Minerals Classification')
    parser.add_argument('--config', type=str, default='config/imagenet/imagenet_san10_pairwise.yaml', help='config file')
    parser.add_argument('opts', help='see config/imagenet/imagenet_san10_pairwise.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    # logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)

    model = SAN()
    # logger.info(model)
    model = torch.nn.DataParallel(model.cuda())
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    
    if os.path.isdir(args.save_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    mean1, std1 = [0.385, 0.372, 0.36], [0.213, 0.215, 0.225]
    mean2, std2 = [0.294, 0.353, 0.37], [0.192, 0.209, 0.227]

    val_domain_folders = [ "/deepo_data/GSP/heavy_minerals/data/Yangtze/val",]
                          # "/deepo_data/GSP/heavy_minerals/data/PumQu/val",
                          # "/deepo_data/GSP/heavy_minerals/data/Yangtze/val"]
    # val_domain_folders = [
    #                       "/deepo_data/GSP/heavy_minerals/data/PumQu/val"
    #                       "/deepo_data/GSP/heavy_minerals/data/Yangtze/val"
    #                       ]
    transform1 = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.RandomRotation(30),transforms.CenterCrop((224, 224)),
         transforms.ToTensor(), transforms.Normalize(mean1, std1)])
    transform2 = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.RandomRotation(30),transforms.CenterCrop((224, 224)),
         transforms.ToTensor(), transforms.Normalize(mean2, std2)])
    val_set = DomainImageFolder(val_domain_folders, transform1=transform1, transform2=transform2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)
    validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    outputs = torch.FloatTensor([])
    targets = torch.FloatTensor([])

    test_time_transforms = tta.Compose(
        [
            # tta.HorizontalFlip(),
            # tta.VerticalFlip(),
            # tta.Rotate90(angles=[0, 90, 180]),
            # tta.Add(values=[i*0.01 for i in range(1, 30)])
        ]
    )
    
    model.eval()
    end = time.time()
    for i, (input, target, domain) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        domain = domain.cuda(non_blocking=True)

        tta_outputs = []

        for transformer in test_time_transforms:
            augmented_image = transformer.augment_image(input)
            with torch.no_grad():
                output, output_domain = model(augmented_image)
            tta_outputs.append(output.detach().cpu().numpy().tolist())

        output = np.mean(tta_outputs, axis=0)
        output = torch.tensor(output).cuda(non_blocking=True)
        loss = criterion(output, target)


        top1, top5 = cal_accuracy(output, target, topk=(1, 5))
        n = input.size(0)
        loss_meter.update(loss.item(), n), top1_meter.update(top1.item(), n), top5_meter.update(top5.item(), n)

        output = output.max(1)[1]

        targets = torch.cat([targets,target.cpu().float()])
        outputs = torch.cat([outputs,output.cpu().float()])
        
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f} '
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f}).'.format(i + 1, len(val_loader),
                                                                        data_time=data_time,
                                                                        batch_time=batch_time,
                                                                        loss_meter=loss_meter,
                                                                        accuracy=accuracy,
                                                                        top1=top1_meter,
                                                                        top5=top5_meter))
    
    #precision = precision_score(targets, outputs, average='weighted')
    #recall = recall_score(targets, outputs, average='weighted')
    #f1 = f1_score(targets, outputs, average='weighted')
    #logger.info('Precision/Recall/F1-score {:.4f}/{:.4f}/{:.4f}'.format(precision, recall, f1))
    results = classification_report(targets, outputs, output_dict=True)
    logger.info(results)
    logger.info("\n{}".format(pd.DataFrame(results)))
    logger.info('kappa:{}'.format(cohen_kappa_score(targets, outputs)))
    logger.info("confusion matrix:\n{}".format(confusion_matrix(targets, outputs)))
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc/top1/top5 {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc, top1_meter.avg, top5_meter.avg))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc, top1_meter.avg, top5_meter.avg


if __name__ == '__main__':
    main()
