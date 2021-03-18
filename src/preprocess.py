"""
    Preprocess input CPC file and output InstanceFeatures

    Authors:
        Anon <anon@anon.anon>
    Date created: March 7, 2021
    Python version: 3.6.0+
"""

import csv
import argparse
from tqdm import tqdm

import gensim
import nltk
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

from utils import InstanceFeatures, Embeddings
from utils import dump_pickle, build_token_to_orig_map, dynamic_padding
from utils import wordpiece2word

from parsers import semeval_14, semeval_15_16
from data_types import Target


DATA_DIR = "./data/"

OOV_TOK = "OutOfVocab"


def convert_Target_to_Instance(tgt:Target, 
    tokenizer:BertTokenizer, task: str, sample_id: int) -> InstanceFeatures:
    """
    Convert Target of ABSA to InstanceFeatures
    """
    target = tgt.get("target")
    sentiment = tgt.get("sentiment") # -1,0,1 ==> 0,1,2 (NEG,NEU,POS)
    text = tgt.get("text")

    tokenize_output = tokenizer(target)
    token_ids, mask = tokenize_output['input_ids'], tokenize_output['attention_mask']

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    sentence_from_tokens = tokenizer.convert_tokens_to_string(tokens)
    token_to_orig_map = build_token_to_orig_map(tokens)

    return InstanceFeatures(
        task=task, sample_id=sample_id, tokens=tokens, entityA=target, entityB=None,
        token_ids=token_ids, token_mask=mask, label=str(sentiment), 
        label_id=sentiment+1, token_to_orig_map=token_to_orig_map,
        sentence=sentence_from_tokens, we_indices=None)


# Return a list of InstanceFeatures. One InstanceFeature for each sentence.
def preprocess_cpc(file_path, bert_version):

    # return values
    cpc_data_features = []

    tokenizer = BertTokenizer.from_pretrained(bert_version)
    label2id = {"BETTER": 0, "WORSE": 1, "NONE": 2}

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)

        for idx, row in tqdm(enumerate(reader)):
            sentence = row['sentence']
            label = row['most_frequent_label']
            entityA = row['object_a']
            entityB = row['object_b']

            tokenizer_output = tokenizer(sentence)

            token_ids = tokenizer_output['input_ids']
            mask = tokenizer_output['attention_mask']

            # keep the special tokens inside the `tokens`
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            sentence_from_tokens = tokenizer.convert_tokens_to_string(tokens)
            token_to_orig_map = build_token_to_orig_map(tokens)

            cpc_data_features.append(
                InstanceFeatures(task="cpc", sample_id=idx, entityA=entityA,
                    entityB=entityB, tokens=tokens, token_ids=token_ids,
                    token_mask=mask, label=label, label_id=label2id[label],
                    token_to_orig_map=token_to_orig_map, 
                    sentence=sentence_from_tokens, we_indices=None))

    return cpc_data_features


def preprocess_bert_embedding(instance_features, bert):
    all_wordlevel_emb = {}

    # do not run the back propagation.
    with torch.no_grad():
        for idx, ins in enumerate(instance_features):
            assert idx == ins.get_sample_id, "[BERT] idx does NOT match sample ID"
            # convert python lists to torch tensors
            tokens_tensor = torch.tensor([ins.get_token_ids()])
            mask_tensors = torch.tensor([ins.get_token_mask()])
            output = bert(tokens_tensor, mask_tensors)
            wp_emb = output.last_hidden_state

            # squeeze the tensor to remove the batch
            wp_emb = torch.squeeze(wp_emb, dim=0)  # (token_len, dim)
            wd_emb = wordpiece2word(wp_emb, ins.get_token_to_orig_map())

            all_wordlevel_emb[idx] = wd_emb

    return all_wordlevel_emb


def preprocess_glove_embedding(instance_features, model):
    def get_embedding(word):
        try:
            word_embedding = model[word]
        except:
            word_embedding = model[OOV_TOK]
        return word_embedding

    all_wordlevel_emb_glove = {}
    for idx, ins in enumerate(instance_features):
        assert idx == ins.sampled_id, "[GLOVE] idx does NOT match sample ID"
        wd_tokens = nltk.word_tokenize(ins.sentence)
        wd_emb = torch.tensor([
            get_embedding(wd) for wd in wd_tokens], dtype=torch.float)
        all_wordlevel_emb_glove[idx] = wd_emb
    
    return all_wordlevel_emb_glove
    

def preprocess_absa():
    """
    Process Aspect-Based Sentiment Analysis datasets of SemEval-14/15/16 Tasks.
    Many thanks to the original author, Henry B. Moss, of the methods `semeval_14` 
    and `semeval_15_16`. The code was cloned from his repo:
        https://github.com/henrymoss/COLING2018/
    """
    all_data_targets = []

    # process SemEval-14 Restuarant Training Dataset
    f_res14 = DATA_DIR + "rawSemEval/SemEval2014/SemEval2014_Restaurants_Train_v2.xml"
    res14_data = semeval_14(f_res14)
    all_data_targets.extend(res14_data.data())

    # process SemEval-15 Restuarant Training Dataset
    f_res15 = DATA_DIR + "rawSemEval/SemEval2015/ABSA-15_Restaurants_Train_Final.xml"
    res15_data = semeval_15_16(f_res15)
    all_data_targets.extend(res15_data.data())

    # process SemEval-16 Restuarant Training Dataset
    f_res16 = DATA_DIR + "rawSemEval/SemEval2016/absa_train.xml"
    res16_data = semeval_15_16(f_res16)
    all_data_targets.extend(res16_data.data())

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    all_data_instances = []
    for idx, target in tqdm(enumerate(all_data_targets)):
        all_data_instances.append(convert_Target_to_Instance(
            target, tokenizer, task="absa", sample_id=idx))
    
    return all_data_instances
    

def preprocess_depgraph(instance_features):
    """
    TODO for Yilong
    Build dependency graphs for input instances

    Args:
        instance_features - list of instance features
    
    Return:
        depg_dict - Dict[idx, depgraph]. 
    """
    all_depgraph = {}
    for idx, ins in instance_features:
        assert idx == ins.sample_id, "[DepParse] idx does NOT match sample ID"
        # TODO: build dep graph
        depg = None 
        all_depgraph[idx] = depg

    return all_depgraph

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_bert_emb", action="store_true", default=False)
    parser.add_argument("--bert_version", type=str, required=False,
        default="bert-base-uncased", help="The version of BERT.")

    parser.add_argument("--generate_glove_emb", action="store_true", default=False)
    parser.add_argument("--glove_dimension", type=int, required=False)

    parser.add_argument("--generate_dep_graph", action="store_true", default=False)

    args = parser.parse_args()

    # preprocess cpc
    print("[preprocess] processing cpc data ...")
    cpc_trn_data = preprocess_cpc(DATA_DIR + "data.csv", args.bert_version)
    cpc_tst_data = preprocess_cpc(DATA_DIR + "hold_out.csv", args.bert_version)

    # preprocess absa
    print("[preprocess] processing absa data ...")
    absa_data = preprocess_absa()

    # dump data
    print("[preprocess] dumping processed cpc/absa instances to {}.".format(DATA_DIR))
    dump_pickle(DATA_DIR+"processed_cpc_train.pkl", cpc_trn_data)
    dump_pickle(DATA_DIR+"processed_cpc_test.pkl", cpc_tst_data)
    dump_pickle(DATA_DIR+"processed_absa.pkl", absa_data)

    # generate bert embedding 
    if args.generate_bert_emb:
        print("[preprocess] generating BERT embedding ...")

        bert = BertModel.from_pretrained(args.bert_version, output_hidden_states=True)
        bert.eval()

        print("\t\t CPC data ...")
        cpc_trn_bert_emb = preprocess_bert_embedding(cpc_trn_data, bert)
        cpc_tst_bert_emb = preprocess_bert_embedding(cpc_tst_data, bert)
        dump_pickle(DATA_DIR+"cpc_train_bert_emb.pkl", cpc_trn_bert_emb)
        dump_pickle(DATA_DIR+"cpc_test_bert_emb.pkl", cpc_tst_bert_emb)

        print("\t\t ABSA data ...")
        absa_bert_emb = preprocess_bert_embedding(absa_data, bert)
        dump_pickle(DATA_DIR+"absa_bert_emb.pkl", absa_bert_emb)

    if args.generate_glove_emb:
        print("[preprocess] generating GLOVE embedding ...")

        glove_path = "./data/glove/glove.6B.{}d.word2vec_format.txt".format(
            args.glove_dimension)
        glove = gensim.models.KeyedVectors.load_word2vec_format(glove_path)
        glove[OOV_TOK] = np.random.rand(args.glove_dimension)

        print("\t\t CPC data ...")
        cpc_trn_glove_emb = preprocess_glove_embedding(cpc_trn_data, glove)
        cpc_tst_glove_emb = preprocess_glove_embedding(cpc_tst_data, glove)
        dump_pickle(DATA_DIR+"cpc_train_glove_emb.pkl", cpc_trn_glove_emb)
        dump_pickle(DATA_DIR+"cpc_test_glove_emb.pkl", cpc_tst_glove_emb)
    
        print("\t\t ABSA data ...")
        absa_glove_emb = preprocess_glove_embedding(absa_data, glove)
        dump_pickle(DATA_DIR+"absa_glove_emb.pkl", absa_glove_emb)

    if args.generate_dep_graph:
        print("[preprocess] generating Dependency Graph ...")
        print("\t\t CPC data ...")
        cpc_trn_depg = preprocess_depgraph(cpc_trn_data)
        cpc_tst_depg = preprocess_depgraph(cpc_tst_data)
        dump_pickle(DATA_DIR+"cpc_train_depgraph.pkl", cpc_trn_depg)
        dump_pickle(DATA_DIR+"cpc_test_depgraph.pkl", cpc_tst_depg)

        print("\t\t ABSA data ...")
        absa_depg = preprocess_depgraph(absa_data)
        dump_pickle(DATA_DIR+"absa_depgraph.pkl", absa_depg)