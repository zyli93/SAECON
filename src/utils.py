import pickle

"""
    Utililty files 

    Authors:
     Zihan Liu <leoliu00529@gmail.com>
"""

def load_pickle(path):
    """ load pickle object from file """
    with open(path, "rb") as fin:
        return pickle.load(fin)


def dump_pickle(path, obj):
    """ dump object to pickle file """
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)

class InstanceFeatures:
    def __init__(self,
                 tokens,
                 token_ids,
                 token_mask,
                 labels,
                 label_ids,
                 token_to_orig_map,
                 sentence,
                 we_indices):
        """
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
        self.tokens     = tokens
        self.token_ids  = token_ids
        self.token_mask = token_mask
        self.labels     = labels
        self.label_ids  = label_ids
        self.token2orig = token_to_orig_map
        self.sentence   = sentence
        self.we_indices = we_indices

    def get_tokens(self):
        return self.tokens

    def get_token_ids(self):
        return self.token_ids

    def get_token_mask(self):
        return self.token_mask

    def get_labels(self):
        return self.labels

    def get_label_ids(self):
        return self.label_ids

    def get_token_to_orig_map(self):
        return self.token2orig

    def get_we_indices(self):
        return self.we_indices


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