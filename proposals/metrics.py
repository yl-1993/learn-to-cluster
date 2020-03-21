#!/usr/bin/env python
# -*- coding: utf-8 -*-


def convert2set(x):
    if isinstance(x, set):
        return x
    elif isinstance(x, list):
        return set(x)
    else:
        return set(list(x))


def compute_iop(pred, label):
    s1 = convert2set(pred)
    s2 = convert2set(label)
    return 1. * len(s1 & s2) / len(s1)


def compute_iog(pred, label):
    s1 = convert2set(pred)
    s2 = convert2set(label)
    return 1. * len(s1 & s2) / len(s2)


def compute_iou(pred, label):
    s1 = convert2set(pred)
    s2 = convert2set(label)
    return 1. * len(s1 & s2) / len(s1 | s2)
