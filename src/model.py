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

# from module import SGCNConv
from ABSA.saecc_train import SAECC_ABSA
from utils import dynamic_padding
from constants import *

class SaeccModel(nn.Module):
    def __init__(self, args):
        # TODO: to be done
        
        pass

    def forward(self, ):
        # TODO: to be done
        pass


class AbsaPipeline(nn.Module):
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = int(batch_size)
        self.absa = SAECC_ABSA(self.batch_size)


    def forward(self, batch):
        # batch_embedding: a list of tensors, each of shape [sentence length, 768]
        # batch_instance_feature: a list of instance features that are in the same order as in batch_embedding
        batch_embedding = batch['embedding']
        batch_instance_feature = batch['instance_feature']

        assert len(batch_embedding) == len(batch_instance_feature), \
            "Number of embedding does not match numberof instance features."
        assert len(batch_embedding) == self.batch_size, "Batch size does not match with batch size."

        # Max sequence length is hardcoded to 80, following original paper setup
        padded_embedding, _ = dynamic_padding(batch_embedding, 80)

        batch_data = []
        for i, each_embedding in enumerate(padded_embedding):
            batch_data.append([each_embedding, batch_instance_feature[i]])

        assert len(batch_data) == self.batch_size, "Final batch input does not match with batch size."

        logits, train_loss = self.absa.run_batch(batch_data)

        return {
            'logits': logits,
            'train_loss': train_loss
        }

    def reset_stats(self):
        # Call this at the start of each epoch to reset the stats used to calculate loss
        self.absa.reset_stats()



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
        self.lstm = nn.LSTM(
            input_size=args.embed_dim,
            hidden_size=args.hidden_dim,
            batch_first=True
        )

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


def FastCpcPipeline(CpcPipeline):
    def _extract_entities(self, batch, node_hidden, word_hidden):
        # TODO: vectorize
        pass
