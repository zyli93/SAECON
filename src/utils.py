import os
import pickle
from torch.nn import ConstantPad2d

import torch
import numpy as np
from datetime import datetime
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score

"""
    Utililty files for SAECC

    Authors:
        Anon <anon@anon.anon>
"""

DATA_DIR = "./data/"
LOG_DIR = "./log/"
CKPT_DIR = "./ckpt/"

REV_LABEL = {
    "BETTER": "WORSE",
    "NONE": "NONE",
    "WORSE": "BETTER"
}

LABEL2ID = {"BETTER": 0, "WORSE": 1, "NONE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

ASBA_LABELS = {"-1": 0, "0": 1, "1": 2}
ID2LABEL_ABSA = {0: "NEG", 1: "NEU", 2: "POS"}
LABELS = [0, 1, 2]

def load_pickle(path):
    """ load pickle object from file """
    with open(path, "rb") as fin:
        return pickle.load(fin)


def dump_pickle(path, obj):
    """ dump object to pickle file """
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)

def make_dir(path):
    """helper for making dir"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_time():
    time = datetime.now().isoformat()[5:24]
    return time

def print_args(args):
    not_print = set([])
    print("\n"+"="*70)
    print("\t Argument Settings")
    for arg in vars(args):
        if arg not in not_print:
            print("\t" + arg + " : " + str(getattr(args, arg)))
    print("="* 70 + "\n")

def get_entity_pos(doc_sentence, doc_entity):
    ls, le = len(doc_sentence), len(doc_entity)
    for i in range(ls):
        if doc_sentence[i:i+le].text == doc_entity.text:
            return list(range(i, i+le))


def pretoken2token(doc, bert_tokenizer):
    token_ids = []
    token_to_orig_map = {}
    wp_idx = 1
    for wd_idx, word in enumerate(doc):
        # bert word tokenization
        tokenizer_output = bert_tokenizer(word.text, add_special_tokens=False)
        token_ids.extend(tokenizer_output['input_ids'])
        # build wp to wd map
        for _ in tokenizer_output['input_ids']:
            token_to_orig_map[wp_idx] = wd_idx
            wp_idx += 1
    token_ids = [101] + token_ids + [102]
    assert len(token_ids) == wp_idx + 1, \
        f"# of wordpieces mismatch {len(token_ids)} vs {wp_idx + 1}"

    return token_ids, token_to_orig_map


class InstanceFeatures:
    def __init__(self,
                 task: str,
                 sample_id: int,
                 entityA: str,
                 entityA_pos: list,
                 entityB: str,
                 entityB_pos: list,
                 tokens: list,
                 token_ids: list,
                 token_mask: list, 
                 label: str,
                 label_id: int,
                 token_to_orig_map: dict,
                 sentence: str,
                 sentence_raw: str,
                 we_indices: list):
        """
        TODO: revise the docstring for new arguments
        Attributes of this class:
            self.tokens: a list of tokens (str) generated by BertTokenizer.
            self.token_ids: a list of the IDs of the self.tokens. Please note 
                that len(token_ids) = len(tokens) + 2 due to the insertions special
                tokens of [CLS] and [SEP].
            self.token_mask: a by-product of BertTokenizer for as the 
                mask input to Bert model.
            self.labels: one of the five labels {BETTER, WORSE, NONE}
            self.label_ids: one ID for each label (0 to 2)
            self.token_to_orig_map: the mapping from positions of token ids to 
                the position of the original word. E.g., if [A, B, C] is tokenized
                to [[CLS], A, B1, B2, B3, C, [SEP]], the mapping is 
                {1:0, 2:1, 3:1, 4:1, 5:2}.
            self.sentence: the tokenized and then `convert_tokens_to_sentence()`-ed
                sentence.
            self.we_indices: Set to None for now.
        """
        self.task           = task
        self.sample_id      = sample_id
        self.tokens         = tokens
        self.entityA        = entityA
        self.entityA_pos    = entityA_pos
        self.entityB        = entityB
        self.entityB_pos    = entityB_pos
        self.token_ids      = token_ids
        self.token_mask     = token_mask
        self.label          = label
        self.label_id       = label_id
        self.token2orig     = token_to_orig_map
        self.sentence       = sentence
        self.sentence_raw   = sentence_raw
        self.we_indices     = we_indices
    
    def get_sample_id(self):
        return self.sample_id
    
    def get_entities(self):
        return self.entityA, self.entityB
    
    def get_entity_positions(self):
        return self.entityA_pos, self.entityB_pos

    def get_tokens(self):
        return self.tokens

    def get_token_ids(self):
        return self.token_ids

    def get_token_mask(self):
        return self.token_mask

    def get_label(self):
        return self.label

    def get_label_id(self):
        return self.label_id

    def get_token_to_orig_map(self):
        return self.token2orig

    def get_we_indices(self):
        return self.we_indices
    
    def get_task(self):
        return self.task
    
    def get_sentence(self):
        return self.sentence
    
    def get_sentence_raw(self):
        return self.sentence_raw


# def convert_tokens_to_sentence(tokens):
#     """convert a list of tokens to a sentences"""
#     # change "'" to "##'"
#     tokens = [token if token != "'" else "##'" for token in tokens]
#     text = ' '.join([x for x in tokens])
#     return text.replace(' ##', '')


class Embeddings:
    def __init__(self, embedding, embedding_without_word_piece):

        """
        This class holds the embedding of a sentence. It is generated by feeding an InstanceFeature object in to
        preprocess_embedding() in src/preprocess.py
        :param embedding: the embedding as is returned from bert without modification
        :param embedding_without_word_piece: the embedding without word piece. Individual word pieces of a word are
        combined and averaged to obtain the embedding for such word
        """
        self.embedding = embedding
        self.embedding_without_word_piece = embedding_without_word_piece

    def get_embedding(self):
        return self.embedding

    def get_embedding_without_word_piece(self):
        return self.embedding_without_word_piece


# def build_token_to_orig_map(tokens):
#     token_indices = list(range(1, len(tokens) - 1))
#     tok2orig_list = []
#     for i in token_indices:
#         if i == 1:
#             tok2orig_list.append(0)
#         else:
#             if len(tokens[i]) > 2 and tokens[i][0:2] == "##":
#                 tok2orig_list.append(tok2orig_list[-1])
#             else:
#                 tok2orig_list.append(tok2orig_list[-1] + 1)
    
#     assert len(token_indices) == len(tok2orig_list), "Unequal lengths!"
#     token_to_orig_map = dict(zip(token_indices, tok2orig_list))
#     return token_to_orig_map

def wordpiece2word(emb, wp2wd, use_gpu):
    """Convert word piece embedding to word embedding

    Args:
        emb - [torch.Tensor] (wp_size, emb_dim) embedding matrix of wordpiece
        wp2wd - wordpiece to word mapping, "token2orig_map"
    
    Var:
        wp_size - the number of wordpieces in the `emb` matrix WITHOUT padding dims
        wd_size - the size of target words (no padding)
    Return:
        norm_mask^T \dot emb in shape of (wd_size, emb_dim)
    """
    wp_size, _= emb.shape[0], emb.shape[1]
    wd_size = max(wp2wd.values()) + 1
    wp_size2 = max(wp2wd.keys()) + 2

    assert wp_size == wp_size2, "emb dim does NOT match wp2wd keys"

    coord_row, coord_col = zip(*wp2wd.items())
    data = np.ones(len(coord_row), dtype=np.float32)
    mask = coo_matrix((data, (coord_row, coord_col)), shape=(wp_size, wd_size)).toarray()
    norm_mask = normalize(mask, norm="l1", axis=0)  # (wp_size * wd_size)
    norm_mask = torch.from_numpy(norm_mask)

    if use_gpu:
        norm_mask = norm_mask.cuda()

    return torch.mm(norm_mask.t(), emb)
  
# This method takes a list of pytorch tensor and returns a list of padded tensors and the mex_length
# Ex: [[53, 768], [12, 768]] -> [[53, 768], [53, 768]], 53

def dynamic_padding(tensor_list, length = None):
    """[not used]"""
    padded_list = []

    if length is not None:
        for tensor in tensor_list:
            pad_len = length - tensor.shape[0]
            padding = ConstantPad2d((0, 0, 0, pad_len), 0)
            padded_list.append(padding(tensor))

        return padded_list, length


    tensor_lengths = [x.shape[0] for x in tensor_list]
    max_length = max(tensor_lengths)

    for tensor in tensor_list:
        pad_len = max_length - tensor.shape[0]
        padding = ConstantPad2d((0, 0, 0, pad_len), 0)
        padded_list.append(padding(tensor))

    return padded_list, max_length


def reverse_instance(ins, sample_id):
    entityB, entityA = ins.get_entities()
    entityB_pos, entityA_pos = ins.get_entity_positions()
    rev_label = REV_LABEL[ins.get_label()]
    return InstanceFeatures(
        task=ins.get_task(), 
        sample_id=sample_id,
        tokens=ins.get_tokens(), 
        entityA=entityA, 
        entityB=entityB, 
        entityA_pos=entityA_pos, 
        entityB_pos=entityB_pos,
        token_ids=ins.get_token_ids(), 
        token_mask=ins.get_token_mask(),
        label=rev_label, 
        label_id=LABEL2ID[rev_label],
        token_to_orig_map=ins.get_token_to_orig_map(), 
        sentence=ins.get_sentence(),
        sentence_raw=ins.get_sentence_raw(), 
        we_indices=None)


def eval_metric(y_true, y_pred, labels):
    each_class_f1 = f1_score(y_true, y_pred, lables=labels, average=None)
    metric_dict = dict(zip(labels, each_class_f1))
    metric_dict["micro"] = f1_score(y_true, y_pred, average=None)
    return metric_dict
