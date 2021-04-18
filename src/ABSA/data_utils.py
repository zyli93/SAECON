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


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


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


def text_to_berttok_seq(tokenizer, text, padlen):
    # sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
    sequence = tokenizer(text)['input_ids']
    if len(sequence) == 0:
        sequence = [0]
    return pad_or_truncate(sequence, padlen)


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


nlp = spacy.load("en_core_web_sm")

def calculate_dep_dist(sentence,aspect):
    """
    compute the smallest distance on dep-graph from any token (tokenized by spaCy)
    in the sentence to the aspect
    """
    # TODO: absa + dependency parsing
    terms = [a.lower() for a in aspect.split()]
    doc = nlp(sentence)
    # Load spacy's dependency tree into a networkx graph
    edges = []
    cnt = 0
    term_ids = [0] * len(terms)
    for token in doc:
        # Record the position of aspect terms
        if cnt < len(terms) and token.lower_ == terms[cnt]:
            term_ids[cnt] = token.i
            cnt += 1

        for child in token.children:
            edges.append(('{}_{}'.format(token.lower_,token.i),
                          '{}_{}'.format(child.lower_,child.i)))

    graph = nx.Graph(edges)

    dist = [0.0]*len(doc)
    text = [0]*len(doc)
    for i,word in enumerate(doc):
        source = '{}_{}'.format(word.lower_,word.i)
        sum = 0
        for term_id,term in zip(term_ids,terms):
            target = '{}_{}'.format(term, term_id)
            try:
                sum += nx.shortest_path_length(graph,source=source,target=target)
            except:
                sum += len(doc) # No connection between source and target
        dist[i] = sum/len(terms)
        text[i] = word.text
    return text,dist