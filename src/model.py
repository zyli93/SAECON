"""
    File for SAECC model

    Authors:
        Anon <anon@anon.anon>

    Date created: March 11, 2020
    Python version: 3.6.0

    # TODO: add index attribute to InstanceFeature
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader as TorchDataLoader

from module import SGCNConv

from ABSA.saecc_train import SAECC_ABSA
from ABSA.models import LCFS_BERT
from ABSA.data_utils import Tokenizer4Pretrain  # TODO: need this?
from ABSA.data_utils import ABSADataset  # TODO: need this?

from utils import dynamic_padding, pad_or_trunc_to_fixlength
from constants import *
from transformers import BertTokenizer, BertModel
from transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig

class SaeccModel(nn.Module):
    def __init__(self, args):
        self().__init__()
        self.cpc_pipeline = CpcPipeline(args)
        self.absa_pipeline = AbsaPipeline(args)

        hidden_dim = self.cpc_pipeline.output_dim + self.absa_pipeline.output_dim
        self.linear = nn.Linear(hidden_dim, 3)

        self._reset_params()

    def forward(self, batch):
        # TODO: Differentiate cpc and absa!
        hidden_cpc = self.cpc_pipeline(batch)
        hidden_absa = self.absa_pipeline(batch)

        hidden_agg = torch.cat(hidden_cpc.values())
        hidden_agg = torch.cat([hidden_agg, hidden_absa['logits']])
        
        logits = nn.linear(hidden_agg)
        return logits

    # TODO: add initialization code!
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


class AbsaPipeline(nn.Module):
    def __init__(self, args):
        super().__init__()

        """
        # TODO: argument to args
        self.batch_size = int(batch_size)
        self.absa = SAECC_ABSA(self.batch_size)

        # TODO: save output_dim for saecc
        self.output_dim = None
        """

        # === New code ===
        # TODO: fix opt:
        #   pretrained_bert_name: bert-based-cased
        #   max_seq_len: (default=80)
        #   input_cols
        # TODO: args: add absa_max_seq_len
        self.max_seq_len = args.absa_max_seq_len
        opt.inputs_cols = ['bert_embedding', 'text_raw_bert_indices', 'aspect_bert_indices',
                           'dep_distance_to_aspect', 'polarity']

        transformer = BertModel.from_pretrained(
            opt.pretrained_bert_name, output_attentions=True)

        tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
        tokenizer = Tokenizer4Pretrain(tokenizer, opt.max_seq_len)
        self.tokenizer = tokenizer
        self.model = LCFS_BERT(transformer, opt)

        # TODO: figure out what's done here.
        _params = filter(lambda p: p.requires_grad, self.model.parameters())


    def forward(self, batch):
        """
        batch_embedding: 
          a list of tensors, each of shape [sentence length, 768]
        batch_instance_feature: 
          a list of instance features that are in the same order as in batch_embedding
        """
        batch_embedding = batch['embedding']
        batch_instance_feature = batch['instance_feature']

        # Max sequence length is hardcoded to 80, following original paper setup
        # TODO: set to 80
        padded_embedding = pad_or_trunc_to_fixlength(
            batch_embedding, length=self.max_seq_len)

        # Now: padded_embedding (batch_size, args.absa_max_seq_len)
        batch_data = []
        for i, each_embedding in enumerate(padded_embedding):
            batch_data.append([each_embedding, batch_instance_feature[i]])

        assert len(batch_data) == self.batch_size, \
            "Final batch input does not match with batch size."

        trainset = ABSADataset(batch_data, self.tokenizer)
        train_data_loader = TorchDataLoader(dataset=trainset, 
            batch_size=self.opt.batch_size, shuffle=False)
        
        # TODO: what's in dataloader?

        for each in train_data_loader:
            # inputs to dictionary
            # inputs = [each[col].to(self.opt.device) for col in self.opt.inputs_cols]
            # TODO: move these stuff to device
            inputs = {k: each[k] for k in self.opt.input_cols}

            # TODO: feature and logits
            outputs = self.model(inputs)
            
            # TODO: collect outputs
        
        inputs = {}  # TODO
        outputs = self.model(inputs)  # TODO: two results

        return outputs  # TODO: dimension of outputs



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
