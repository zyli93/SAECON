# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

# Heavily modified by Zihan Liu and Zeyu Li

import os
import pickle
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import networkx as nx
import spacy
from utils import InstanceFeatures
from transformers import BertTokenizer
from torch.utils.rnn import pad_sequence


def pad_or_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

def pad_or_truncate_tensorlist(tensor_list, length):
    """pad or truncate a list of tensors"""
    assert isinstance(tensor_list[0], torch.Tensor), "tesnor type not right"
    padded = pad_sequence(tensor_list, batch_first=True)
    if padded.shape[1] > length:
        padded = padded[:, :length]
    elif padded.shape[1] < length:
        pad_len = length - padded.shape[1]
        tensor_to_pad = torch.zeros(padded.shape[0], pad_len)
        padded = torch.cat([padded, tensor_to_pad], axis=1)
    return padded

def pad_to_fixedlength(tensor, length):
    """pad or truncate a list of tensors"""
    if padded.shape[1] > length:
        padded = padded[:, :length]
    elif padded.shape[1] < length:
        pad_len = length - padded.shape[1]
        tensor_to_pad = torch.zeros(padded.shape[0], pad_len)
        padded = torch.cat([padded, tensor_to_pad], axis=1)
    return padded


def retok_with_dist(tokenizer, text, dep_dist, padlen):
    """
    convert spacy tokenization to bert tokenization
    assign distance of each spacy-token to resultant bert-token
    """
    distances = []
    for word, dist in zip(text, dep_dist):
        tokens = tokenizer.tokenize(word)
        distances += [dist] * len(tokens)

    if len(distances) == 0:
        distances = [0]
    token_dist = pad_or_truncate(distances, padlen)
    return token_dist
