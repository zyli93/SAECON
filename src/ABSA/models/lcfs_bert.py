# -*- coding: utf-8 -*-
# file: lcfs_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn
import copy
import numpy as np

from transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig

class PointwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid=None,d_out=None, dropout=0):
        super(PointwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        if d_out is None:
            d_out = d_inner_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output

class SelfAttention(nn.Module):
    def __init__(self, config,opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(
            np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len), 
                dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out, att = self.SA(inputs, zero_tensor)

        SA_out = self.tanh(SA_out)
        return SA_out, att

"""
    attributes in opt to be moved to args:
        1. dropout
        2. local_context_focus
"""
class LCFS_BERT(nn.Module):
    def __init__(self, args, use_gpu):
        super(LCFS_BERT, self).__init__()
        
        self.hidden, hidden = args.emb_dim, args.emb_dim
        sa_config = BertConfig(hidden_size=self.hidden, output_attentions=True)
        self.use_gpu = use_gpu
        self.local_context_focus = args.absa_local_context_focus

        self.dropout = nn.Dropout(args.absa_dropout)
        self.bert_sa = SelfAttention(sa_config, args.absa_max_seq_len)  
        # TODO: why bert SelfAttention???

        # self.mean_pooling_double = nn.Linear(hidden * 2, hidden)
        self.mean_pooling_double = PointwiseFeedForward(hidden * 2, hidden,hidden)
        self.bert_pooler = BertPooler(sa_config)
        self.dense = nn.Linear(hidden, 3)  # Hard-coded polarities-dim = 3

    def feature_dynamic_mask(self, text_local_indices, aspect_indices, distances_input=None):
        # TODO: why there are cpu?
        texts = text_local_indices.cpu().numpy() # batch_size x seq_len
        asps = aspect_indices.cpu().numpy() # batch_size x aspect_len
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.hidden),
                                          dtype=np.float32) # batch_size x seq_len x hidden size
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))): # For each sample
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i]) # Calculate aspect length
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][0])[0][0]
                except:
                    continue
                # Mask begin -> Relative position of an aspect vs the mask
                if asp_begin >= mask_len:
                    mask_begin = asp_begin - mask_len
                else:
                    mask_begin = 0
                for i in range(mask_begin): # Masking to the left
                    masked_text_raw_indices[text_i][i] = np.zeros((self.hidden), dtype=np.float)
                for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len): # Masking to the right
                    masked_text_raw_indices[text_i][j] = np.zeros((self.hidden), dtype=np.float)
            else:
                distances_i = distances_input[text_i]
                for i,dist in enumerate(distances_i):
                    if dist > mask_len:
                        masked_text_raw_indices[text_i][i] = np.zeros((self.hidden), dtype=np.float)

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        if self.use_gpu:
            masked_text_raw_indices = masked_text_raw_indices.cuda()
        return masked_text_raw_indices
        # TODO: fix devices issues

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices,distances_input=None):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        masked_text_raw_indices = np.ones(
            (text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
            dtype=np.float32) # batch x seq x dim
        mask_len = self.opt.SRD
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i]) - 2
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][2])[0][0]
                    asp_avg_index = (asp_begin * 2 + asp_len) / 2 # central position
                except:
                    continue
                distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
                for i in range(1, np.count_nonzero(texts[text_i])-1):
                    srd = abs(i - asp_avg_index) + asp_len / 2
                    if srd > self.opt.SRD:
                        distances[i] = 1 - (srd - self.opt.SRD)/np.count_nonzero(texts[text_i])
                    else:
                        distances[i] = 1
                for i in range(len(distances)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
            else:
                distances_i = distances_input[text_i] # distances of batch i-th
                for i,dist in enumerate(distances_i):
                    if dist > mask_len:
                        distances_i[i] = 1 - (dist - mask_len) / np.count_nonzero(texts[text_i])
                    else:
                        distances_i[i] = 1

                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances_i[i]

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        if self.use_gpu:
            masked_text_raw_indices = masked_text_raw_indices.cuda()
        return masked_text_raw_indices
        # TODO: fix devices issues


    def forward(self, inputs, output_attentions = False):
        bert_embedding = inputs["bert_embedding"]
        text_local_indices = inputs['text_raw_bert_indices']
        aspect_indices = inputs['aspect_bert_indices']
        distances = inputs['dep_distance_to_aspect']
        bert_local_out = bert_embedding

        if self.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(
                text_local_indices, 
                aspect_indices, 
                distances)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(
                text_local_indices, 
                aspect_indices, 
                distances)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        out_cat = torch.cat((bert_local_out, bert_embedding), dim=-1)
        mean_pool = self.mean_pooling_double(out_cat)
        self_attention_out, local_att = self.bert_sa(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)
        return dense_out

        # TODO: opt.local_context_focus
        # TODO: locate feature and logits