# -*- coding: utf-8 -*-
# file: lcfs_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn
import copy
import numpy as np

from transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig

from ABSA.data_utils import pad_or_truncate_tensorlist, text_to_berttok_seq
from ABSA.data_utils import retok_with_dist
from ABSA.data_utils import pad_or_truncate
from transformers import BertTokenizer

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
    def __init__(self, config, max_seq_len, device):
        super(SelfAttention, self).__init__()
        self.config = config
        self.SA = BertSelfAttention(config)

        self.tanh = torch.nn.Tanh()
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, inputs):
        # zero_tensor is used as attention_mask
        zero_tensor = torch.tensor(
            np.zeros((inputs.size(0), 1, 1, self.max_seq_len), 
                dtype=np.float32), dtype=torch.float32)
        zero_tensor = zero_tensor.to(self.device)
        SA_out, att = self.SA(inputs, zero_tensor)

        SA_out = self.tanh(SA_out)
        return SA_out, att

"""
    attributes in opt to be moved to args:
        1. dropout
        2. local_context_focus
"""
class LCFS_BERT(nn.Module):
    def __init__(self, args, device):
        super(LCFS_BERT, self).__init__()
        
        self.hidden, hidden = args.emb_dim, args.emb_dim
        self.device= device
        self.local_context_focus = args.absa_local_context_focus
        self.max_seq_len = args.absa_max_seq_len
        self.bert_dim = args.emb_dim
        self.SRD = args.absa_syntactic_relative_distance

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_version)
        self.dropout = nn.Dropout(args.absa_dropout)

        sa_config = BertConfig(hidden_size=self.hidden, output_attentions=True)
        self.bert_sa = SelfAttention(sa_config, args.absa_max_seq_len, device)  

        self.mean_pooling_double = PointwiseFeedForward(hidden * 2, hidden,hidden)
        self.bert_pooler = BertPooler(sa_config)
        self.readout = nn.Linear(hidden, 3)  # Hard-coded polarities-dim = 3
        self.output_dim = hidden

    def feature_dynamic_mask(self, text_local_indices, distances_input=None):
        texts = text_local_indices
        mask_len = self.SRD

        # masked_text_raw_indices (batch_size, self.max_seq_len, self.hidden)
        masked_text_raw_indices = np.ones(
            (len(text_local_indices), self.max_seq_len, self.hidden),
            dtype=np.float32) # batch_size x seq_len x hidden size
        for text_i in range(len(texts)):
            distances_i = distances_input[text_i]
            for i,dist in enumerate(distances_i):
                if dist > mask_len:
                    masked_text_raw_indices[text_i][i] = np.zeros(
                        (self.hidden), dtype=np.float)

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        masked_text_raw_indices = masked_text_raw_indices.to(self.device)
        return masked_text_raw_indices
        

    def feature_dynamic_weighted(self, text_local_indices, distances_input=None):
        texts = text_local_indices
        masked_text_raw_indices = np.ones(
            (len(text_local_indices), self.max_seq_len, self.bert_dim),
            dtype=np.float32) # batch x seq x dim
        mask_len = self.SRD
        for text_i in range(len(text_local_indices)):
            distances_i = distances_input[text_i] # distances of batch i-th
            for i,dist in enumerate(distances_i):
                if dist > mask_len:
                    distances_i[i] = 1 - (dist - mask_len) / np.count_nonzero(texts[text_i])
                else:
                    distances_i[i] = 1

            for i in range(len(distances_i)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances_i[i]

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        masked_text_raw_indices = masked_text_raw_indices.to(self.device)
        return masked_text_raw_indices


    def forward(self, original_batch, switch=False):

        # convert original batch to new batch
        inputs = self.convert_batch_to_absa_batch(original_batch, switch)

        bert_embedding = inputs["bert_embedding"]
        text_local_indices = inputs['text_raw_bert_indices']
        distances = inputs['dep_distance_to_aspect']
        bert_local_out = bert_embedding

        if self.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(
                text_local_indices, 
                distances)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(
                text_local_indices, 
                distances)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        out_cat = torch.cat((bert_local_out, bert_embedding), dim=-1)
        mean_pool = self.mean_pooling_double(out_cat)
        self_attention_out, local_att = self.bert_sa(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        prediction = self.readout(pooled_out)
        return pooled_out, prediction
    
    def convert_batch_to_absa_batch(self, original_batch, switch):
        """Convert original batch to absa batch. In detail, we need to prepare
        a few things: embedding, depdency-based distance to aspect term, bert tokenizations.

        Args:
            original_batch - A batch from the dataloader
            switch - Specifically for ABSA batchs, switch entityA and entityB

        """
        ent = "aspdistB" if switch else "aspdistA"

        emb = original_batch['embedding']
        ins_feat = original_batch['instance_feature']
        assert len(emb) == len(ins_feat), "Num of emb doesn't match num of ins_feat"

        padded_emb = pad_or_truncate_tensorlist(emb, self.max_seq_len)  # (batch_size, absa_fix_len)

        text_raw_bert_indices = [ins.get_token_ids() 
            for ins in original_batch['instances']]
        token_dist_list = original_batch[ent] 
        
        absa_batch = {
            "bert_embedding": padded_emb,  # Padded, tensor
            "text_raw_bert_indices": text_raw_bert_indices,  # variable length
            "dep_distance_to_aspect": token_dist_list  # variable length
        }
        return absa_batch