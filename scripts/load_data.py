# -*- coding: utf-8 -*-
"""
Created on May 13, 2020
@author: yongzhengxin
"""
import numpy as np
import pickle
from nltk.corpus import framenet as fn

def load_data(fn_lu_embedding_filename, fn_plus_lu_embedding_filename):
    fn_L = pickle.load(open(fn_lu_embedding_filename, 'rb'))
    fn_plus_L = pickle.load(open(fn_plus_lu_embedding_filename, 'rb'))

    frames_to_int = {}
    for lu_id in fn_L.keys():
        frame_name = fn.lu(lu_id).frame.name
        if frame_name not in frames_to_int:
            frames_to_int[frame_name] = len(frames_to_int)

    X = list()
    Y = list()

    for lu_id, tensor in fn_L.items():
        frame_name = fn.lu(lu_id).frame.name
        X.append(tensor.numpy())
        Y.append(frames_to_int[frame_name])

    cut_off = len(Y)

    for keys, tensor in fn_plus_L.items():
        frame_name, new_lu, ori_lu = keys
        X.append(tensor.numpy())
        Y.append(frames_to_int[frame_name])
    return np.array(X), np.array(Y), len(frames_to_int), cut_off


def load_anomalous_synsets(anomalous_synset_embedding_filename):
    W = pickle.load(open(anomalous_synset_embedding_filename, 'rb'))
    anom_X = list()
    anom_Y = list()

    for synset_name, tensor in W.items():
        anom_X.append(tensor.numpy())
        anom_Y.append(synset_name)

    import random
    random.seed(123)
    np.random.seed(123)
    for i in range(len(anom_X)):
        # (700, 0.6) - gput40
        for _ in range(700):
            col = random.randint(0, 3071)
            anom_X[i][col] = np.random.uniform(0.3, 0.9)
            print(anom_X[i][col])

    # # high precision
    # for i in range(len(anom_X)):
    #     if random.random() < 0.01:
    #         anom_X[i] = np.random.random(3072)
    pickle.dump([np.array(anom_X), np.array(anom_Y)], open("corrupted_anom.p", 'wb'))
    return np.array(anom_X), np.array(anom_Y)
