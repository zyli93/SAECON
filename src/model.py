"""
    File for SAECC model

    Authors:
        Anon <anon@anon.anon>

    Date created: March 11, 2020
    Python version: 3.6.0

    # TODO: add index attribute to InstanceFeature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from module import SGCNConv
from constants import *

class SaeccModel(nn.Module):
    def __init__(self, args):
        # TODO: to be done
        
        pass

    def forward(self, ):
        # TODO: to be done
        pass


class AbsaPipeline(nn.Module):
    def __init__(self, args):
        pass



class CpcPipeline(nn.Module):
    def __init__(self, args):
        # global context
        sgcn_convs = []
        sgcn_dims = [args.embed_dim] + args.sgcn_dims
        for d_in, d_out in zip(sgcn_dims[:-1], sgcn_dims[1:]):
            sgcn_convs.append(
                SGCNConv(
                    dim_in=d_in,
                    dim_out=d_out,
                    num_labels=len(DEPENDENCY_LABELS),
                    gating=args.sgcn_gating
                )
            )
        self.sgcn_convs = nn.Sequential(*sgcn_convs)
        
        # local context
        self.lstm = nn.LSTM(
            input_size=args.embed_dim,
            hidden_size=args.hidden_dim,
            batch_first=True
        )
    
    def forward(self, batch):
        # global context
        depgraph = batch['depgraph']
        depgraph.x = self.sgcn_convs(
            x=depgraph.x,
            edge_index=depgraph.edge_index,
            edge_label=depgraph.edge_attr
        )
        node_hidden = pad_sequence(
            [dg.x for dg in depgraph.to_data_list()],
            batch_first=True
        )

        # local context
        word_embedding = batch['embedding']
        word_hidden = self.lstm(word_embedding)

        assert node_hidden.shape[0] == word_hidden.shape[0], 'batch size do not match'
        assert node_hidden.shape[1] == word_hidden.shape[1], 'seq_len do not match'

        # extract entities
        instances = batch['instances']
        entA_pos = [torch.tensor(ins.entityA_pos) for ins in instances]
        entB_pos = [torch.tensor(ins.entityB_pos) for ins in instances]

        nodeA, nodeB = [], []
        for seq, posA, posB in zip(node_hidden, entA_pos, entB_pos):
            embedA = torch.index_select(seq, 0, posA)
            nodeA.append(torch.mean(embedA, dim=0))
            embedB = torch.index_select(seq, 0, posB)
            nodeB.append(torch.mean(embedB, dim=0))

        wordA, wordB = [], []
        for seq, posA, posB in zip(word_hidden, entA_pos, entB_pos):
            embedA = torch.index_select(seq, 0, posA)
            wordA.append(torch.mean(embedA, dim=0))
            embedB = torch.index_select(seq, 0, posB)
            wordB.append(torch.mean(embedB, dim=0))
        
        return {
            'nodeA': torch.cat(nodeA),
            'nodeB': torch.cat(nodeB),
            'wordA': torch.cat(wordA),
            'wordB': torch.cat(wordB)
        }