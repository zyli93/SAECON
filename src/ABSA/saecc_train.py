import argparse
import math
import torch
import torch.nn as nn
from pytorch_transformers import BertModel
from torch.utils.data import DataLoader
from pytorch_transformers import BertTokenizer
from ABSA.data_utils import ABSADataset, Tokenizer4Pretrain
from ABSA.models import LCFS_BERT
from ABSA.models.aen import CrossEntropyLoss_LSR


class SAECC_ABSA:
    def __init__(self, batch_size):

        # Initialize params
        parser = argparse.ArgumentParser()
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--l2reg', default=0.01, type=float)
        parser.add_argument('--embed_dim', default=300, type=int)
        parser.add_argument('--hidden_dim', default=300, type=int)
        parser.add_argument('--bert_dim', default=768, type=int)
        parser.add_argument('--pretrained_bert_name', default='bert-base-cased', type=str)
        parser.add_argument('--max_seq_len', default=80, type=int)
        parser.add_argument('--polarities_dim', default=3, type=int)
        parser.add_argument('--hops', default=3, type=int)
        parser.add_argument('--lsr', default=False)

        parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
        parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
        parser.add_argument('--valset_ratio', default=0, type=float,
                            help='set ratio between 0 and 1 for validation support')
        parser.add_argument('--local_context_focus', default='cdw', type=str,
                            help='local context focus mode, cdw or cdm')
        parser.add_argument('--SRD', default=4, type=int, help='set SRD')
        opt = parser.parse_args()
        opt.batch_size = batch_size
        opt.model_class = LCFS_BERT
        opt.inputs_cols = ['bert_embedding', 'text_raw_bert_indices', 'aspect_bert_indices',
                           'dep_distance_to_aspect', 'polarity']
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.opt = opt

        # Initialize model
        tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
        transformer = BertModel.from_pretrained(opt.pretrained_bert_name, output_attentions=True)
        tokenizer = Tokenizer4Pretrain(tokenizer, opt.max_seq_len)
        self.tokenizer = tokenizer
        
        # TODO: do we really need transformer?
        self.model = LCFS_BERT(transformer, opt).to(opt.device)

        self.model.train()

        self.n_correct = 0
        self.n_total = 0
        self.loss_total = 0

    def reset_stats(self):
        self.n_correct = self.n_total = self.loss_total = 0

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

    def run_batch(self, batch_data):

        """

        :param batch_data: A list of batch_size elements. Each elements in the form of [embedding, instanceFeature]
        :return:
        """

        # TODO: better keep this 
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        trainset = ABSADataset(batch_data, self.tokenizer)
        train_data_loader = DataLoader(dataset=trainset, 
            batch_size=self.opt.batch_size, shuffle=False)

        for each in train_data_loader:
            inputs = [each[col].to(self.opt.device) for col in self.opt.inputs_cols]
            outputs = self.model(inputs)
            targets = each['polarity'].to(self.opt.device)
