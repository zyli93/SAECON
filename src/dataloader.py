"""
    The file for dataloader of SAECC

    Authors:
        Anon <anon@anon.anon>

    Date created: March 11, 2020
    Python version: 3.6.0

    TODO:
    1. preprocess sentence embedding for cpc and absa
"""

import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch

from utils import DATA_DIR
from utils import load_pickle
from utils import reverse_instance

CPC = "CPC"
ABSA = "ABSA"
VAL_RATIO = 0.2

class DataLoader():
    def __init__(self, args):
        self.batch_size = args.batch_size

        # Batch Ordering
        # return_seq is a list of two integers, which will be verified later
        #   return_seq[0] is the number of CPC iterations,
        #   return_seq[1] is the number of ABSA iterations.
        # this variable will be used in self.get_train_batch()
        self.return_seq = [int(x) for x in args.batch_ratio.split(":")]
        if len(self.return_seq) != 2:
            raise ValueError("batch ratio should have two values.")

        self.batch_counter = 0

        ##########################################################
        # Variables
        # - cpc_trn and cpc_tst: CPC training & test datasets.
        # - cpc_trn_emb and cpc_tst_emb: CPC embedding for training 
        #       and for test. When "ft" is used, these vars are None.
        # - cpc_trn_depg and cpc_tst_depg: CPC dependency graph
        #       for training and test.
        # - absa: ABSA training datasets.
        # - absa_emb: ABSA sentence embedding for training.
        # 
        # Note:
        #   1. validation are drawn from train
        #   2. If the current logic runs too slow, consider fix the t/v/t split
        #       esp. the validation.
        #   3. Training indices: self.cpc_trn_indices, self.absa
        ##########################################################

        self.cpc_trn, self.cpc_tst = None, None
        self.cpc_trn_emb, self.cpc_tst_emb = None, None
        self.cpc_trn_depg, self.cpc_tst_depg = None, None

        self.absa = None
        self.absa_emb = None
        self.absa_len = None

        # check & process input embedding type
        if args.input_emb not in ["ft", "fix", "glove"]:
            raise ValueError("Invalid value of input_emb!")
        fine_tune = True if args.input_emb == "ft" else False
        emb_model = "glove" if args.input_emb == "glove" else "bert"

        # will need to load these either ways
        self.cpc_trn = load_pickle(DATA_DIR+"processed_cpc_train.pkl")
        self.cpc_tst = load_pickle(DATA_DIR+"processed_cpc_test.pkl")
        self.absa = load_pickle(DATA_DIR+"processed_absa.pkl")
        self.cpc_trn_depg = load_pickle(DATA_DIR+"cpc_trn_depg.pkl")
        self.cpc_tst_depg = load_pickle(DATA_DIR+"cpc_tst_depg.pkl")

        self.absa_len = len(self.absa)

        if not fine_tune:
            print("[DataLoader] loading data from disk ...")
            self.cpc_trn_emb = load_pickle(
                DATA_DIR+"cpc_train_{}_emb.pkl".format(emb_model))
            self.cpc_tst_emb = load_pickle(
                DATA_DIR+"cpc_test_{}_emb.pkl".format(emb_model))
            self.absa_emb = load_pickle(
                DATA_DIR+"absa_{}_emb.pkl".format(emb_model))

        # Dynamically separate 20% in each label for validation
        self.cpc_trn_indices, self.cpc_val_indices = self.__split_val_for_cpc()
        self.cpc_tst_indices = list(range(len(self.cpc_tst)))
        
        if args.data_augmentation:
            print("[DataLoader] augmenting data ...")
            self.__data_augmentation(fine_tune)
        
        print("[DataLoader] Data loader loading done!")
        self.trn_batch_num = self.__get_batch_num(len(self.cpc_trn_indices), self.batch_size)
        self.tst_batch_num = self.__get_batch_num(len(self.cpc_tst_indices), self.batch_size)
        self.val_batch_num = self.__get_batch_num(len(self.cpc_val_indices), self.batch_size)

    
    def __data_augmentation(self, fine_tune):
        id_ = len(self.cpc_trn)
        new_indices = []
        for ins in self.cpc_trn_indices:
            # reverse a single training instance 
            if ins.get_label() != "NONE":
                ins_id = ins.get_sample_id()
                rev_ins = reverse_instance(ins, sample_id=id_)
                self.cpc_trn.append(rev_ins)
                self.cpc_trn_depg[id_] = self.cpc_trn_depg[ins_id]
                if not fine_tune:
                    self.cpc_trn_emb[id_] = self.cpc_trn_emb[ins_id]
                new_indices.append(id_)
                id_ += 1
        self.cpc_trn_indices.extend(new_indices)


    def __get_batch_num(self, ds, bs):
        """ds: data size; bs: batch size"""
        tail_batch = 1 if ds % bs else 0 
        return ds // bs + tail_batch


    def __fetch_batch(self, task, split, indices):
        """fetch a batch from a list of indices
        
        Args:
            ins_feats - [List] of indices
        Return:
            batch - [Dict] dictionary of batch
        """
        if task == CPC:
            # CPC train/validation
            if split == "train" or "val":
                instances = [self.cpc_trn[x] for x in indices]
                emb = [self.cpc_trn_emb[x] for x in indices]
                depg = [self.cpc_trn_depg[x] for x in indices]
            # CPC test
            else:
                instances = [self.cpc_tst[x] for x in indices]
                emb = [self.cpc_tst_emb[x] for x in indices]
                depg = [self.cpc_tst_depg[x] for x in indices]
        else:
            instances = [self.absa[x] for x in indices]
            emb = [self.absa_emb[x] for x in indices]
            depg = None

        # batch dependency graph
        depg_list = [
            Data(
                x=emb_i, 
                edge_index=depg_i['edge_index'], 
                edge_attr=depg_i['edge_label']
            )
            for emb_i, depg_i in zip(emb, depg)
        ]
        depg = Batch.from_data_list(depg_list)
        
        emb = pad_sequence(emb, batch_first=True)
        return {"task": task, "instances": instances,
            "embedding": emb, "depgraph": depg}
    

    def __split_val_for_cpc(self):
        indices = {i:[] for i in [0,1,2]}
        trn_indices, val_indices = [], []
        trn_counts, val_counts = [], []
        for ins in self.cpc_trn:
            indices[ins.get_label_id()].append(ins.get_sample_id)
        for label in [0,1,2]:
            trn_idx, val_idx = train_test_split(indices[label], VAL_RATIO)
            trn_indices.extend(trn_idx)
            val_indices.extend(val_idx)
            trn_counts.append(len(trn_idx))
            val_counts.append(len(val_idx))
        
        print("[DataLoader] dynamically splitting trn to trn+val ...")
        print("\t[Train] 0:{}, 1:{}, 2:{}; [Validation] 0:{}, 1:{}, 2:{}".format(
            *trn_counts, *val_counts))
        
        return trn_indices, val_indices

    
    def get_batch_train(self):
        bs = self.batch_size
        for ptr in range(self.trn_batch_num):
            end = min((ptr+1)*bs, len(self.cpc_trn_indices))
            indices = self.cpc_trn_indices[ptr*bs: end]
            yield self.__fetch_batch(CPC, "train", indices)
            if ptr and ptr % self.return_seq[0] == 0:
                for _ in range(self.return_seq[1]):
                    indices = list(np.random.choice(range(self.absa_len), bs))
                    yield self.__fetch_batch(ABSA, "train", indices)

        self.batch_counter += 1
        # dict: task cpc/absa, bert emb, label, depg
    

    def get_batch_testval(self, for_test=True):
        """get data iterator for test and validation"""
        bs = self.batch_size
        data_indices = self.cpc_tst_indices if for_test else self.cpc_val_indices
        split = "test" if for_test else "val"
        for ptr in range(self.tst_batch_num):
            end = min((ptr+1)*bs, len(data_indices))
            indices = data_indices[ptr*bs: end]
            yield self.__fetch_batch(CPC, split, indices)