import torch

from evaluation import precision, recall, accuracy


def online_evaluate(gtmat, pred):
    pred_labels = torch.argmax(pred.cpu(), dim=1).long()
    gt_labels = gtmat.view(-1).cpu().numpy()
    pred_labels = pred_labels.numpy()
    acc = accuracy(gt_labels, pred_labels)
    pre = precision(gt_labels, pred_labels)
    rec = recall(gt_labels, pred_labels)
    return acc, pre, rec
