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
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

from module import SGCNConv
from ABSA.models import LCFS_BERT
from constants import *
from utils import get_activ

class SaeccModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.use_dom_inv = args.dom_adapt  # whether to use domain invariant
        self.device = device
        self.cpc_pipeline = CpcPipeline(args, device)
        self.absa_pipeline = LCFS_BERT(args, device)

        # cpc_pipeline.output_dim = `feature_dim` for word + `feature_dim` for node
        # absa_pipeline.output_dim = feature_dim
        pipeline_output_dim = self.cpc_pipeline.output_dim + self.absa_pipeline.output_dim

        # CPC readout layer
        self.cpc_readout = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(2*pipeline_output_dim, pipeline_output_dim)),
          ('activ1', get_activ(args.activation)),
          ('dropout', nn.Dropout(args.dropout)),
          ('linear2', nn.Linear(pipeline_output_dim, 3))
        ]))

        dom_inv_dim = self.absa_pipeline.output_dim
        if args.dom_adapt:
            self.dom_inv = nn.Sequential(OrderedDict([
                ('revgrad', RevGrad()), 
                ('linear', nn.Linear(dom_inv_dim, 1)),
                ('activ', get_activ('relu'))
            ]))

        self._reset_params()

    def forward(self, batch):
        all_entA = all_entB = []
        # CPC. cpc_pipeline outputs a dict of `nodeA`, `nodeB`, `wordA`, and `wordB`
        if batch['task'] == CPC:
            hidden_cpc = self.cpc_pipeline(batch)
            hidden_absa_entA, entA = self.absa_pipeline(batch)
<<<<<<< HEAD
            hidden_absa_entB, entB = self.absa_pipeline(batch, switch=True)

=======
            # print(entA.size())
            hidden_absa_entB, entB = self.absa_pipeline(batch, switch=True)
            all_entA.append(entA)
            all_entB.append(entB)
>>>>>>> b62679a... debug
            # After cat: (batch_size, 3*feature_dim)
            hidden_entA = torch.cat(
                [hidden_cpc['nodeA'], hidden_cpc['wordA'], hidden_absa_entA], 1)
            hidden_entB = torch.cat(
                [hidden_cpc['nodeB'], hidden_cpc['wordB'], hidden_absa_entB], 1)
            
            # CPC readout
            pred = self.cpc_readout(
                torch.cat([hidden_entA, hidden_entB], dim=1))
            
            # hidden_absa: (2*batch_size, feature_dim)
            hidden_absa = torch.cat([hidden_absa_entA, hidden_absa_entB])

        # ABSA. absa_pipeline outputs: pooled_out, prediction
        else:
            hidden_absa, pred = self.absa_pipeline(batch)
        
        if self.use_dom_inv:
            dom_logit = self.dom_inv(hidden_absa)
            return {'prediction': pred, "domain_logit": dom_logit, "entityA": all_entA, "entityB": all_entB}

        if batch['task'] == CPC:
            return {'prediction': pred, 'entityA': entA, 'entityB': entB}
        else:
            return {'prediction': pred}


    def _reset_params(self):
        initializer = torch.nn.init.xavier_normal_
        for child in self.children():
            # print(child)
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
    def __init__(self, args, device):
        super().__init__()
        """
        #   emb_dim -> sgcn_dim0, 
        #   sgcn_dim0 -> sgcn_dim1 -> ... -> sgcn_dim[-1]
        #   sgcn_dim[-1] -> feature_dim
        """
        self.device = device
        # global context
        sgcn_dims = [args.emb_dim]+ args.sgcn_dims + [args.feature_dim]

        # TODO: from children printout: params not properly registered
        # self.sgcn_convs = [
        #     SGCNConv(
        #         dim_in=d_in,
        #         dim_out=d_out,
        #         num_labels=len(DEPENDENCY_LABELS),
        #         gating=args.sgcn_gating
        #     )
        #     for d_in, d_out in zip(sgcn_dims[:-1], sgcn_dims[1:])
        # ]

        self.sgcn_convs = nn.ModuleList([
            SGCNConv(
                dim_in=d_in,
                dim_out=d_out,
                num_labels=len(DEPENDENCY_LABELS),
                gating=args.sgcn_gating,
                directed=args.sgcn_directed
            )
            for d_in, d_out in zip(sgcn_dims[:-1], sgcn_dims[1:])
        ])


        # local context
        # (batch, seq_len, 2*hidden)
        self.lstm = nn.LSTM(
            input_size=args.emb_dim,
            hidden_size=args.feature_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.output_dim = args.feature_dim * 2

    def forward(self, batch):
        # global context
        depgraph = batch['depgraph']
        # depgraph: Batch
        # depgraph.edge_index: Tensor (on device)
        # depgraph.edge_attr: Tensor (on device)

        # TODO: verify this part w/ Yilong
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
            'nodeA': torch.vstack(nodeA),
            'nodeB': torch.vstack(nodeB),
            'wordA': torch.vstack(wordA),
            'wordB': torch.vstack(wordB)
        }

    def _extract_entities(self, batch, node_hidden, word_hidden):
        # extract entities
        instances = batch['instances']
        entA_pos = [torch.tensor(ins.entityA_pos).to(self.device) for ins in instances]
        entB_pos = [torch.tensor(ins.entityB_pos).to(self.device) for ins in instances]

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

        return nodeA, nodeB, wordA, wordB


def FastCpcPipeline(CpcPipeline):
    def _extract_entities(self, batch, node_hidden, word_hidden):
        # TODO (for Yilong): vectorize (what's this?)

        pass

class EDGAT(nn.Module):
    def __init__(
        self, 
        dim_in,
        n_layers,
        device
        ):
        super().__init__()
        self.device = device
        dims = [dim_in] + [300] * n_layers
        self.gat_convs = nn.ModuleList([
            GATConv(d_in, d_out // 6, heads=6)
            for d_in, d_out in zip(dims[:-1], dims[1:])
        ])
        self.readout = nn.Linear(dims[-1] * 2, 3)
    
    def forward(self, batch):
        depgraph = batch['depgraph']
        depgraph.edge_index = to_undirected(depgraph.edge_index)
        for conv in self.gat_convs:
            depgraph.x = conv(
                x=depgraph.x,
                edge_index=depgraph.edge_index
            )

        node_hidden = pad_sequence(
            [dg.x for dg in depgraph.to_data_list()],
            batch_first=True
        )
        
        nodeA, nodeB = self._extract_entities(batch, node_hidden)
        nodeA, nodeB = torch.vstack(nodeA), torch.vstack(nodeB) 
        nodes = torch.cat([nodeA, nodeB], 1)

        logits = self.readout(nodes)
        return logits

    def _extract_entities(self, batch, node_hidden):
        # extract entities
        instances = batch['instances']
        entA_pos = [torch.tensor(ins.entityA_pos).to(self.device) for ins in instances]
        entB_pos = [torch.tensor(ins.entityB_pos).to(self.device) for ins in instances]

        nodeA, nodeB = [], []
        for seq, posA, posB in zip(node_hidden, entA_pos, entB_pos):
            embedA = torch.index_select(seq, 0, posA)
            nodeA.append(torch.mean(embedA, dim=0))
            embedB = torch.index_select(seq, 0, posB)
            nodeB.append(torch.mean(embedB, dim=0))

        return nodeA, nodeB
