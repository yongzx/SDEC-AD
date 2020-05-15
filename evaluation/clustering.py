# -*- coding: utf-8 -*-
"""
Created on Nov 04, 2019
@author: yongzhengxin
"""

import numpy as np
from sklearn import metrics
import bcubed

def purity_score(y_true, y_pred, inv=False):
    """
    :param y_true: true cluster ids
    :param y_pred: predicted cluster ids
    :param inv: boolean
    :return: purity (inv = False) or inverse-purity (inv = True)
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    axis = 0 if not inv else 1

    # return purity
    return np.sum(np.amax(contingency_matrix, axis=axis)) / np.sum(contingency_matrix)


def f_purity_score(y_true, y_pred):
    """
    :param y_true: true cluster ids
    :param y_pred: predicted cluster ids
    :return: F1 purity score

    Implementation details - harmonic mean of purity and inverse purity score - see https://arxiv.org/pdf/1401.4590.pdf
    """
    return 2 * (purity_score(y_true, y_pred) * purity_score(y_true, y_pred, inv=True)) / (purity_score(y_true, y_pred) + purity_score(y_true, y_pred, inv=True))


def external_eval_clusters(y_true, y_pred):
    """
    :param y_true: true cluster ids
    :param y_pred: predicted cluster ids
    :return: external evaluation metrics of clustering quality.
    The metrics are purity, inverse purity, harmonic mean, b-cubed precision, recall and their harmonic mean.
    """
    purity = purity_score(y_true, y_pred)
    inverse_purity = purity_score(y_true, y_pred, inv=True)
    f_purity = f_purity_score(y_true, y_pred)

    ldict = {i: {cluster_idx} for i, cluster_idx in enumerate(y_true)}
    cdict = {i: {cluster_idx} for i, cluster_idx in enumerate(y_pred)}
    bcubed_precision = bcubed.precision(cdict, ldict)
    bcubed_recall = bcubed.recall(cdict, ldict)
    bcubed_fscore = bcubed.fscore(bcubed_precision, bcubed_recall)

    return purity, inverse_purity, f_purity, bcubed_precision, bcubed_recall, bcubed_fscore


def print_external_eval_clusters(purity, inverse_purity, f_purity, bcubed_precision, bcubed_recall, bcubed_fscore):
    """
    Print out the external evaluation metrics of clustering quality.
    """
    print("Purity:", purity)
    print("Inverse Purity:", inverse_purity)
    print("F-score (Purity and Inverse Purity):", f_purity)
    print("BCubed Precision:", bcubed_precision)
    print("BCubed Recall:", bcubed_recall)
    print("BCubed F1:", bcubed_fscore)
    return
