"""
    The file for dataloader of SAECC

    Authors:
        Anon <anon@anon.anon>

    Date created: March 11, 2020
    Python version: 3.6.0

    TODO:
    1. preprocess sentence embedding for cpc and absa
"""

from utils import DATA_DIR
from utils import load_pickle

class DataLoader():
    def __init__(self, args):
        """TODO"""
        # TODO: initializing attributes

        # TODO: loading data
        self.cpc_data = self.__load_cpc_data()
        self.absa_data = self.__load_absa_data()

        # TODO: more
        if args.input_emb not in ["ft", "fix", "glove"]:
            raise ValueError("Invalid value of input_emb!")

        if args.input_emb != "ft":
            self.cpc_sent_emb = self.__load_sent_emb(args.input_emb, "cpc")
            self.absa_sent_emb = self.__load_sent_emb(args.input_emb, "absa")
        else:
            self.cpc_sent_emb, self.absa_sent_emb = None, None


    def __load_cpc_data(self):
        pass
        # TODO
    
    def __load_absa_data(self):
        pass
        # TODO
    
    def __load_sent_emb(self, input_emb, task):
        """load sentence embedding
        
        Args:
            input_emb - training input embedding, `ft`, `fix`, or `glove`
            task - learning task, `cpc` and `absa`
        """
        in_file = DATA_DIR + "{}_{}.pkl".format(task, input_emb)
        return load_pickle(in_file)
    
    def get_absa_iterator():
        # TODO
        pass