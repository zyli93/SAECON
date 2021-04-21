"""
    File for SAECC model

    Authors:
        Anon <anon@anon.anon>

    Date created: March 11, 2020
    Python version: 3.6.0

"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
from pytorch_revgrad import RevGrad

from module import SGCNConv
from ABSA.models import LCFS_BERT
from constants import *
from utils import get_activ

class SaeccModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        # Whether to use domain invariant 
        self.use_dom_inv = args.dom_adapt
        self.device = device
        self.cpc_pipeline = CpcPipeline(args)
        self.absa_pipeline = LCFS_BERT(args, device)

        pipeline_output_dim = self.cpc_pipeline.output_dim + self.absa_pipeline.output_dim
        readout_hidden_dim = pipeline_output_dim // 2

        # CPC readout layer
        self.cpc_readout = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(pipeline_output_dim, readout_hidden_dim)),
          ('activ1', get_activ(args.activation)),
          ('dropout', nn.Dropout(args.dropout)),
          ('linear2', nn.Linear(readout_hidden_dim, 3))
        ]))

        dom_inv_dim = self.cpc_pipeline.output_dim
        if args.dom_adapt:
            self.dom_inv = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(dom_inv_dim, 2)),
                ('activ', get_activ('relu')),
                ('revgrad', RevGrad())  # TODO: rev grad position right?
            ]))

        self._reset_params()

    def forward(self, batch):
        # TODO: check on the dimensions

        # CPC. cpc_pipeline outputs a dict of `nodeA`, `nodeB`, `wordA`, and `wordB`
        if batch.task == CPC:
            hidden_cpc = self.cpc_pipeline(batch)
            hidden_absa_entA, _ = self.absa_pipeline(batch)
            hidden_absa_entB, _ = self.absa_pipeline(batch, switch=True)

            hidden_entA = torch.cat(
                [hidden_cpc['nodeA'], hidden_cpc['wordA'], hidden_absa_entA], 1)
            hidden_entB = torch.cat(
                [hidden_cpc['nodeB'], hidden_cpc['wordB'], hidden_absa_entB], 1)
            
            # CPC readout
            pred = self.cpc_readout(
                torch.cat([hidden_entA, hidden_entB], dim=1))
            hidden_absa = torch.cat([hidden_absa_entA, hidden_absa_entB])

        # ABSA. absa_pipeline outputs: pooled_out, prediction
        else:
            hidden_absa, pred = self.absa_pipeline(batch)
        
        if self.use_dom_inv:
            dom_pred = self.dom_inv(hidden_absa)
            return {'prediction': pred, "domain_prediction": dom_pred}

        return {'prediction': pred}


    def _reset_params(self):
        initializer = torch.nn.init.xavier_normal
        for child in self.model.children():
            if type(child) == BertModel:  # skip bert params
                continue
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)


class CpcPipeline(nn.Module):
    def __init__(self, args):
        super().__init__()
        # global context
        sgcn_convs = []
        sgcn_dims = [args.embed_dim] + args.sgcn_dims
        self.sgcn_convs = [
            SGCNConv(
                dim_in=d_in,
                dim_out=d_out,
                num_labels=len(DEPENDENCY_LABELS),
                gating=args.sgcn_gating
            )
            for d_in, d_out in zip(sgcn_dims[:-1], sgcn_dims[1:])
        ]

        # local context
        # (batch, seq_len, 2*hidden)
        self.lstm = nn.LSTM(
            input_size=args.embed_dim,
            hidden_size=args.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.output_dim = (args.hidden_dim + args.sgcn_dims) * 2

    def forward(self, batch):
        # global context
        depgraph = batch['depgraph']
        for conv in self.sgcn_convs:
            depgraph.x = conv(
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
        word_hidden = self.lstm(word_embedding)[0] 
        bs, sl = word_hidden.size(0), word_hidden.size(1)
        word_hidden = torch.mean(word_hidden.view(bs, sl, 2, -1), dim=2)

        assert node_hidden.shape[0] == word_hidden.shape[0], 'batch size do not match'
        assert node_hidden.shape[1] == word_hidden.shape[1], 'seq_len do not match'

        nodeA, nodeB, wordA, wordB = self._extract_entities(batch, node_hidden, word_hidden)

        return {
            'nodeA': torch.cat(nodeA),
            'nodeB': torch.cat(nodeB),
            'wordA': torch.cat(wordA),
            'wordB': torch.cat(wordB)
        }

    def _extract_entities(self, batch, node_hidden, word_hidden):
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

        # TODO (for Yilong): add returns here


def FastCpcPipeline(CpcPipeline):
    def _extract_entities(self, batch, node_hidden, word_hidden):
        # TODO (for Yilong): vectorize (what's this?)

        pass
