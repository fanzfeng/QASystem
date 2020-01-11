# -*- coding: utf-8 -*-
'''
 from https://github.com/faneshion/MatchZoo.git
 可拆分为两部分评价指标
 - 总体评价，常见的如分类问题的p、r、acc
 - 分块评价，例如排序中的p@k，r@k，ndcg@k
 '''

from __future__ import print_function

import random
import numpy as np
import math


def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]


def mse(y_true, y_pred, rel_threshold=0.):
    s = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    return np.mean(np.square(y_pred - y_true), axis=-1)


def accuracy(y_true, y_pred):
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    assert y_true_idx.shape == y_pred_idx.shape
    return 1.0 * np.sum(y_true_idx == y_pred_idx) / len(y_true)


def map(y_true, y_pred, rel_threshold=0):
    s = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = list(zip(y_true, y_pred))
    random.shuffle(c)
    c = sorted(c, key=lambda x: x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / (j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s


def mrr(y_true, y_pred, rel_threshold = 0.):
    k = 10
    s = 0.
    return s

# compute evaluation@k
# the input is all documents under a single query


def ndcg(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.):
        if k <= 0.:
            return 0.
        s = 0.
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c_g = sorted(c, key=lambda x: x[0], reverse=True) # max
        c_p = sorted(c, key=lambda x: x[1], reverse=True) # cur
        idcg = 0.
        ndcg = 0.
        for i, (g,p) in enumerate(c_g):
            if i >= k:
                break
            if g > rel_threshold:
                idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        for i, (g,p) in enumerate(c_p):
            if i >= k:
                break
            if g > rel_threshold:
                ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        if idcg == 0.:
            return 0.
        else:
            return ndcg / idcg
    return top_k


def precision(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.):
        if k <= 0:
            return 0.
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True) # 按得分倒序排列
        prec = 0.
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                prec += 1
                break
        prec /= k
        return prec
    return top_k


def recall(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.):
        if k <= 0:
            return 0.
        y_true = _to_list(np.squeeze(y_true).tolist()) # y_true: the ground truth scores for documents under a query
        y_pred = _to_list(np.squeeze(y_pred).tolist()) # y_pred: the predicted scores for documents under a query
        pos_count = sum(i > rel_threshold for i in y_true) # total number of positive documents under this query
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        recall = 0.
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                recall += 1
        recall /= pos_count
        return recall
    return top_k
# QA问答中top@k中命中了，即为准确


def qa_precision(k=10):
    def top_k(y_true, y_pred):
        if k <= 0:
            return 0.
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = list(zip(y_true, y_pred))
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g >= 1 and p >= 0.5:
                return 1
        return 0
    return top_k
