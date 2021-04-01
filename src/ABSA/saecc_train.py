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
        parser.add_argument('--model_name', default='lcfs_bert', type=str)
        parser.add_argument('--dataset', default='saecc', type=str, help='twitter, restaurant, laptop')
        parser.add_argument('--optimizer', default='adam', type=str)
        parser.add_argument('--initializer', default='xavier_uniform_', type=str)
        parser.add_argument('--learning_rate', default=2e-5, type=float,
                            help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--l2reg', default=0.01, type=float)
        parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
        parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
        parser.add_argument('--log_step', default=5, type=int)
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
        opt.initializer = torch.nn.init.xavier_normal
        opt.optimizer = torch.optim.Adam
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.opt = opt

        # Initialize model
        tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
        transformer = BertModel.from_pretrained(opt.pretrained_bert_name, output_attentions=True)
        tokenizer = Tokenizer4Pretrain(tokenizer, opt.max_seq_len)
        self.tokenizer = tokenizer
        self.model = opt.model_class(transformer, opt).to(opt.device)

        self.model.train()

        self.n_correct = 0
        self.n_total = 0
        self.loss_total = 0


    # def _train(self, criterion, optimizer, train_data_loader):
    #     global_step = 0
    #     n_correct, n_total, loss_total = 0, 0, 0
    #
    #     self.model.train()
    #
    #     for i_batch, sample_batched in enumerate(train_data_loader):
    #         print(i_batch)
    #         print(type(sample_batched))
    #         global_step += 1
    #         # clear gradient accumulators
    #         optimizer.zero_grad()
    #
    #         inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
    #         outputs = self.model(inputs)
    #         targets = sample_batched['polarity'].to(self.opt.device)
    #
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #
    #         n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
    #         n_total += len(outputs)
    #         loss_total += loss.item() * len(outputs)
    #         train_loss = loss_total / n_total
    #
    #         yield (outputs, train_loss * 100)

    def _train_batch(self, criterion, optimizer, sample_batched):

        # clear gradient accumulators
        optimizer.zero_grad()

        inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
        outputs = self.model(inputs)
        targets = sample_batched['polarity'].to(self.opt.device)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        self.n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        self.n_total += len(outputs)
        self.loss_total += loss.item() * len(outputs)
        train_loss = self.loss_total / self.n_total
        train_acc = self.n_correct / self.n_total
        print('loss: {:.4f}, acc: {:.4f}'.format(train_loss * 100, train_acc * 100))
        return outputs, train_loss * 100

    def reset_stats(self):
        self.n_correct = self.n_total = self.loss_total = 0

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    # def run(self):
    #     # Loss and Optimizer
    #     criterion = nn.CrossEntropyLoss()
    #     if self.opt.lsr:
    #         criterion = CrossEntropyLoss_LSR(self.opt.device)
    #     _params = filter(lambda p: p.requires_grad, self.model.parameters())
    #     optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
    #
    #     train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
    #
    #     self._reset_params()
    #
    #     for each_batch in self._train(criterion, optimizer, train_data_loader):
    #         yield each_batch


    def run_batch(self, batch_data):

        """

        :param batch_data: A list of batch_size elements. Each elements in the form of [embedding, instanceFeature]
        :return:
        """

        criterion = nn.CrossEntropyLoss()
        if self.opt.lsr:
            criterion = CrossEntropyLoss_LSR(self.opt.device)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        trainset = ABSADataset(batch_data, self.tokenizer)
        train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=False)

        for each in train_data_loader:
            return self._train_batch(criterion, optimizer, each)







