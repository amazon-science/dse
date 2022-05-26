import os
import csv
import time
import sys


from tqdm import tqdm
import numpy as np
import torch


def test_intent_retrieval(all_labels, embeddings):
    all_labels = np.array(all_labels)

    count = {}
    NUMS = [5, 10, 20, 50, 100]
    for num in NUMS:
        count[num] = 0

    sim_matrix = embeddings @ embeddings.T

    for num in NUMS:
        top = np.argpartition(sim_matrix, -num, axis=1)[:, -num:]
        preds = all_labels[top]
        expand_labels = np.repeat(np.expand_dims(all_labels, 1), num, axis=1)
        count[num] += np.sum(preds == expand_labels) / len(all_labels)

    return count


def test_response_selection(q_embeddings, r_embeddings):
    logits = q_embeddings @ r_embeddings.T
    predictions = np.argsort(logits, axis=1)

    recall_1 = 0
    recall_3 = 0
    recall_5 = 0
    recall_10 = 0

    for i, pred in enumerate(predictions):
        recall_1 += int(i in pred[-1:])
        recall_3 += int(i in pred[-3:])
        recall_5 += int(i in pred[-5:])
        recall_10 += int(i in pred[-10:])
    
    return recall_1, recall_3, recall_5, recall_10


def test_oos(labels, preds, oos_idx=150):
    acc = (preds == labels).mean()
    oos_labels, oos_preds = [], []
    ins_labels, ins_preds = [], []
    for i in range(len(preds)):
        if labels[i] != oos_idx:
            ins_preds.append(preds[i])
            ins_labels.append(labels[i])

        oos_labels.append(int(labels[i] == oos_idx))
        oos_preds.append(int(preds[i] == oos_idx))

    ins_preds = np.array(ins_preds)
    ins_labels = np.array(ins_labels)
    oos_preds = np.array(oos_preds)
    oos_labels = np.array(oos_labels)
    ins_acc = (ins_preds == ins_labels).mean()
    oos_acc = (oos_preds == oos_labels).mean()

    # for oos samples recall = tp / (tp + fn) 
    TP = (oos_labels & oos_preds).sum()
    FN = ((oos_labels - oos_preds) > 0).sum()
    oos_recall = TP / (TP+FN)

    return acc, ins_acc, oos_acc, oos_recall


