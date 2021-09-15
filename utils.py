import os
import numpy as np
import shutil
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

'''
The evaluation metrics code is based on: https://github.com/ZitongYu/CDCN/blob/master/CVPR2020_paper_codes/utils.py
'''

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0/batch_size))
        return res

def get_err_threhold_cross_db(fpr, tpr, threshold):

    RightIndex=(tpr+(1-fpr)-1);
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    return err, best_th, right_index

def performances_cross_db(prediction_scores, gt_labels):

    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    fpr,tpr,threshold = roc_curve(gt_labels, prediction_scores, pos_label=1)

    val_err, val_threshold, right_index = get_err_threhold_cross_db(fpr, tpr, threshold)
    test_auc = auc(fpr, tpr)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    FRR = 1- tpr    # FRR = 1 - TPR

    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate

    return test_auc, fpr[right_index], FRR[right_index], HTER[right_index]

def get_err_threhold(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1);
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]
    return err, best_th

def evalute_performances(prediction_scores, gt_labels):

    fpr_test,tpr_test,threshold_test = roc_curve(gt_labels, prediction_scores, pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, threshold_test)
    test_auc = auc(fpr_test, tpr_test)

    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= best_test_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return best_test_threshold, test_threshold_APCER, test_threshold_BPCER, test_threshold_ACER, test_auc

def evalute_threshold_based(prediction_scores, gt_labels, threshold):
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return test_threshold_APCER, test_threshold_BPCER, test_threshold_ACER
