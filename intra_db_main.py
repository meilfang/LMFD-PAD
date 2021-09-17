from __future__ import print_function, division
import argparse
import os
import random

from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from utils import AvgrageMeter, evalute_performances, evalute_threshold_based
from model.mfad import FAD_HAM_Net
from losses import WeightedFocalLoss

from dataset import FacePAD_Train, FacePAD_Val


# This inter_db.py is used for inter dataset training and evaluation
# The evaluation metrics for cross-domain and inter-dataset is different.

def main():
    # create log file
    log_id = 'pretrain_{}'.format(args.pretrain) + '_lr_{}'.format(args.lr) + '_' + args.prefix

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_path = os.path.join(args.log_dir, log_id + '.txt')

    # for save output prediction labels
    results_dir = os.path.join('results', log_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # create directory for save trained models
    checkpoint_save_dir = os.path.join('checkpoints', log_id)
    best_weights_path = os.path.join(checkpoint_save_dir, 'best_weights.pth')
    if not os.path.isdir(checkpoint_save_dir):
        os.makedirs(checkpoint_save_dir)

    # initialize model
    model = FAD_HAM_Net(pretrain=args.pretrain, variant=args.backbone).cuda()
    print('-------------- train ------------------------')

    log_file = open(os.path.join(args.log_dir, log_id + '.txt'), 'w')

    # load data
    train_path = os.path.join(args.protocol_dir, 'train.csv')
    val_path = os.path.join(args.protocol_dir, 'dev.csv')
    test_path = os.path.join(args.protocol_dir, 'test.csv')

    train_dataset = FacePAD_Train(train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works, pin_memory=True)

    val_dataset = FacePAD_Val(val_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works, pin_memory=True)

    test_dataset = FacePAD_Val(test_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
    smoothl1_criterion = torch.nn.SmoothL1Loss().cuda()
    FL_criterion = WeightedFocalLoss(alpha=.5, gamma=2).cuda()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    scaler = GradScaler()

    # parameters for early stopping
    epochs_no_improvement = 0
    max_auc = -1
    min_acer = 1000
    best_weights = None
    best_epoch = -1

    for epoch in range(1, args.epochs+1):

        loss_total = AvgrageMeter()
        loss_1_total = AvgrageMeter()
        loss_2_total = AvgrageMeter()

        ###########################################
        '''                train             '''
        ###########################################
        model.train()

        # loss weight update
        if epoch >= 5:
            w1 = 100 # 10-75
        else:
            w1 = 1

        progress_bar = tqdm(train_loader)
        for i, (images, labels, map_x) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()
            map_x = map_x.cuda()
            model.zero_grad()

            with autocast():
                pred, map_y = model(images)
                pred = torch.squeeze(pred)
                loss_1 = FL_criterion(pred.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor))
                loss_2 = smoothl1_criterion(map_y, map_x) * w1
                loss = loss_1 + loss_2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_total.update(loss.data, images.shape[0])
            loss_1_total.update(loss_1.data, images.shape[0])
            loss_2_total.update(loss_2.data, images.shape[0])

            progress_bar.set_postfix(
                loss_1 ='%.5f' % (loss_1_total.avg),
                loss_2 ='%.5f' % (loss_2_total.avg),
                loss ='%.5f' % (loss_total.avg),
            )

        ###########################################
        '''                val             '''
        ###########################################
        print ('------------ val -------------------')
        model.eval()
        predictions, gt_labels, video_ids = [], [], []
        with torch.no_grad():
            for (images, labels, _, video_id) in tqdm(val_loader):

                images = images.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                pred, _ = model(images)

                for j in range(images.shape[0]):
                    predictions.append(pred[j].detach().cpu())
                    gt_labels.append(labels[j].detach().cpu())
                    video_ids.append(video_id[j])

        # fuse prediction scores (mean value) of all frames for each video
        predictions, gt_labels, _ = compute_video_score(video_ids, predictions, gt_labels)

        val_th, val_apcer, val_bpcer, val_acer, val_auc = evalute_performances(predictions, gt_labels)
        scheduler.step()

        # check if need early stopping
        if val_acer < min_acer:
            min_acer = val_acer
            epochs_no_improvement = 0
            best_weights = model.state_dict()
            best_epoch = epoch
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= args.patience:
            print(f"EARLY STOPPING at {best_epoch}: {min_acer}")
            break

        tqdm.write('Epoch: %d, Train: loss_total= %.4f, Val: th=%.4f, acer=%.4f \n' % (epoch, loss_total.avg, val_th, val_acer))
        log_file.write('Epoch: %d, Train: loss_total= %.4f, loss_1 = %.4f, loss_2=%.4f, Val: th=%.4f, acer=%.4f, apcer=%.4f, bpcer=%.4f \n' % (epoch, loss_total.avg, loss_1_total.avg, loss_2_total.avg, val_th, val_acer, val_apcer, val_bpcer))
        log_file.flush()

        ###########################################
        '''                Test             '''
        ###########################################
        print ('------------ test -------------------')
        model.eval()
        predictions, gt_labels, video_ids = [], [], []
        with torch.no_grad():
            for (images, labels, _, video_id) in tqdm(test_loader):

                images = images.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                with autocast():
                    pred, _ = model(images)

                for j in range(images.shape[0]):
                    predictions.append(pred[j].detach().cpu())
                    gt_labels.append(labels[j].detach().cpu())
                    video_ids.append(video_id[j])

        predictions, gt_labels, _ = compute_video_score(video_ids, predictions, gt_labels)

        # The threshold from development set is used for evaluate test set.
        test_apcer, test_bpcer, test_acer  = evalute_threshold_based(predictions, gt_labels, val_th)

        tqdm.write('Test based on val_TH: apcer=%.4f, bpcer= %.4f, acer= %.4f \n' % (test_apcer, test_bpcer, test_acer))
        log_file.write('Test based on val_TH: apcer=%.4f, bpcer= %.4f, acer= %.4f  \n' % (test_apcer, test_bpcer, test_acer))
        log_file.flush()

    torch.save(best_weights, best_weights_path)


def compute_video_score(video_ids, predictions, labels):
    predictions_dict, labels_dict = defaultdict(list), defaultdict(list)

    for i in range(len(video_ids)):
        video_key = video_ids[i]
        predictions_dict[video_key].append(predictions[i])
        labels_dict[video_key].append(labels[i])

    new_predictions, new_labels, new_video_ids = [], [], []

    for video_indx in list(set(video_ids)):
        new_video_ids.append(video_indx)
        scores = np.mean(predictions_dict[video_indx])
        label = labels_dict[video_indx][0]
        new_predictions.append(scores)
        new_labels.append(label)

    return new_predictions, new_labels, new_video_ids


if __name__ == "__main__":

    torch.cuda.empty_cache()
    cudnn.benchmark = True

    if torch.cuda.is_available():
        print('GPU is available')
        torch.cuda.manual_seed(0)
    else:
        print('GPU is not available')
        torch.manual_seed(0)

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--protocol_dir', type=str, required=True)
    parser.add_argument('--prefix', default='inter-dataset', type=str)
    parser.add_argument("--backbone", default='resnet50', type=str, choices=['resnet101', 'resnet50', 'resnet34', 'resnet18'])
    parser.add_argument("--pretrain", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']))

    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--epochs", default=100, type=int, help="maximum epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="train batch size")
    parser.add_argument("--num_works", default=32, type=int, help="train batch size")
    parser.add_argument("--patience", default=15, type=int)
    parser.add_argument('--log_dir', type=str, default="training_logs")

    args = parser.parse_args()

    main()
