"""
    Preprocess input CPC file and output InstanceFeatures

    Authors:
        Anon <anon@anon.anon>
    Date created: March 7, 2021
    Python version: 3.6.0+
    
    Below is an example command to run this file:
    ```
    python src/preprocess.py --gpu_id 1 --process_cpc_instances \
        --process_absa_instances --generate_bert_emb --generate_glove_emb \
        --generate_dep_graph

    python src/preprocess.py --generate_dep_graph
    ```
"""

import sys
import re
import csv
import argparse
from tqdm import tqdm

import gensim
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import en_core_web_trf
import networkx as nx

from constants import *
from utils import InstanceFeatures #, Embeddings
from utils import dump_pickle, load_pickle
from utils import pretoken2token
from utils import wordpiece2word, get_entity_pos
from utils import LABEL2ID

from parsers import semeval_14, semeval_15_16
from data_types import Target


DATA_DIR = "./data/"

OOV_TOK = "OutOfVocab"


def convert_Target_to_Instance(tgt:Target, bert_tokenizer, pretokenizer, 
        task, sample_id) -> InstanceFeatures:
    """
    Convert Target of ABSA to InstanceFeatures
    """
    target = tgt.get("target")
    sentiment = tgt.get("sentiment") # -1,0,1 ==> 0,1,2 (NEG,NEU,POS)
    text = tgt.get("text")

    doc = pretokenizer(text)
    pretokens = [tok.text for tok in doc]
    token_ids, token_to_orig_map = pretoken2token(doc, bert_tokenizer)
    mask = [1] * len(token_ids)

    tokens = bert_tokenizer.convert_ids_to_tokens(token_ids)
    sentence_from_tokens = bert_tokenizer.convert_tokens_to_string(tokens)
    entityA_pos = get_entity_pos(doc, pretokenizer(target))

    if not entityA_pos:
        return None

    return InstanceFeatures(
        task=task, sample_id=sample_id, pretokens=pretokens, tokens=tokens, 
        entityA=target, entityB=None,
        entityA_pos=entityA_pos, entityB_pos=None,
        token_ids=token_ids, token_mask=mask, label=str(sentiment), 
        label_id=sentiment+1, token_to_orig_map=token_to_orig_map,
        sentence=sentence_from_tokens, sentence_raw=text, we_indices=None)


# Return a list of InstanceFeatures. One InstanceFeature for each sentence.
def preprocess_cpc(file_path, bert_tokenizer, pretokenizer):

    def sep_punkt(s):
        """separate tokens by slashes, e.g., "apple/banana" -> "apple / banana" """
        s = s.replace("-", " ")
        s = re.sub('([.,!?()-+/])', r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        return s
    
    def merge_dash(s):
        return s.replace("-", " ")

    # return values
    cpc_data_features = []

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)

        index_counter = 0

        for idx, row in tqdm(enumerate(reader)):
            sentence = sep_punkt(row['sentence'])

            label = row['most_frequent_label']
            entityA = merge_dash(row['object_a'])
            entityB = merge_dash(row['object_b'])

            # spacy pretokenization
            doc = pretokenizer(sentence)
            pretokens = [tok.text for tok in doc]
    
            # get bert tokens and token to original word map
            token_ids, token_to_orig_map = pretoken2token(doc, bert_tokenizer)
            mask = [1] * len(token_ids)

            # keep the special tokens inside the `tokens`
            tokens = bert_tokenizer.convert_ids_to_tokens(token_ids)
            sentence_from_tokens = bert_tokenizer.convert_tokens_to_string(tokens)

            # get entity positions
            entityA_pos = get_entity_pos(doc, pretokenizer(entityA))
            entityB_pos = get_entity_pos(doc, pretokenizer(entityB))
            
            if not entityA_pos or not entityB_pos:
                continue

            cpc_data_features.append(
                InstanceFeatures(
                    task=CPC, sample_id=index_counter, 
                    entityA=entityA, entityA_pos=entityA_pos, 
                    entityB=entityB, entityB_pos=entityB_pos,
                    pretokens=pretokens, tokens=tokens, token_ids=token_ids,
                    token_mask=mask, label=label, label_id=LABEL2ID[label],
                    token_to_orig_map=token_to_orig_map, 
                    sentence=sentence_from_tokens, sentence_raw=sentence,
                    we_indices=None
                ))
            index_counter += 1

    return cpc_data_features


def preprocess_bert_embedding(instance_features, bert, use_gpu):
    all_wordlevel_emb = {}

    # do not run the back propagation.
    with torch.no_grad():
        for idx, ins in tqdm(enumerate(instance_features)):
            assert idx == ins.get_sample_id(), "[BERT] idx does NOT match sample ID"
            # convert python lists to torch tensors
            tokens_tensor = torch.tensor([ins.get_token_ids()])
            mask_tensors = torch.tensor([ins.get_token_mask()])
            if use_gpu:
                tokens_tensor = tokens_tensor.cuda()
                mask_tensors = mask_tensors.cuda()

            output = bert(tokens_tensor, mask_tensors)
            wp_emb = output.last_hidden_state

            # squeeze the tensor to remove the batch
            wp_emb = torch.squeeze(wp_emb, dim=0)  # (token_len, dim)
            wd_emb = wordpiece2word(
                wp_emb, ins.get_token_to_orig_map(), use_gpu) # (pretoken, dim)

            all_wordlevel_emb[idx] = wd_emb
        
    if use_gpu:
        """convert gpu tensors to cpu"""
        for key in all_wordlevel_emb.keys():
            all_wordlevel_emb[key] = all_wordlevel_emb[key].cpu()

    return all_wordlevel_emb


def preprocess_glove_embedding(instance_features, model, tokenizer):
    """using the tokenizer from spaCy"""
    def get_embedding(word):
        try:
            word_embedding = model[word]
        except:
            word_embedding = model[OOV_TOK]
        return word_embedding

    all_wordlevel_emb_glove = {}

    for idx, ins in tqdm(enumerate(instance_features)):
        assert idx == ins.get_sample_id(), "[GLOVE] idx does NOT match sample ID"
        wd_tokens = tokenizer(ins.sentence_raw)
        # each wd is a spacy token, therefore use ".text" to convert to str
        wd_emb = torch.tensor([
            get_embedding(wd.text) for wd in wd_tokens], dtype=torch.float)
        all_wordlevel_emb_glove[idx] = wd_emb
    
    return all_wordlevel_emb_glove
    

def preprocess_absa(bert_tokenizer, pretokenizer):
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

    all_data_instances = []
    index_counter = 0
    for idx, target in tqdm(enumerate(all_data_targets)):
        instance = convert_Target_to_Instance(
            target, bert_tokenizer, pretokenizer, task=ABSA, sample_id=index_counter)
        if instance:
            all_data_instances.append(instance)
            index_counter += 1
    
    return all_data_instances
    

def preprocess_depgraph(instance_features, lang_parser):
    """
    Build dependency graphs for input instances
    Note: In spacy 3.0, The lemmatizer is a separate component. Please refer to:
        https://stackoverflow.com/questions/66451577/warning-w108-the-rule-based-lemmatizer-did-not-find-pos-annotation-for-the-to

    Args:
        instance_features - list of instance features
    Return:
        depg_dict - Dict[idx, depgraph]. 
    """
    all_depgraph = {}

    sentences = [ins.sentence_raw for ins in instance_features]
    docs = lang_parser.pipe(sentences, disable=['tagger', 'ner', 'lemmatizer'])

    for idx, doc in tqdm(enumerate(docs)):
        edge_index = []
        edge_label = []

        for token in doc:
            for child in token.children:
                edge_index.append([token.i, child.i])
                edge_label.append(DEPENDENCY_LABELS[child.dep_])

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = torch.t(edge_index).contiguous()
        edge_label = torch.tensor(edge_label, dtype=torch.long)
        all_depgraph[idx] = {
            'edge_index': edge_index,
            'edge_label': edge_label
        }
    
    return all_depgraph


def preprocess_aspect_dist(instance_features, depgraphs, entity):
    assert len(instance_features) == len(depgraphs)
    n_instances = len(instance_features)
    
    all_dists = {}
    for i in tqdm(range(n_instances)):
        ins = instance_features[i]
        depg = depgraphs[i]
        assert i == ins.get_sample_id(), "[ASPECT] idx does NOT match sample ID"

        edges = depg['edge_index']
        edges = torch.t(edges).tolist()
        aspect_pos = getattr(ins, f"entity{entity}_pos")
        n_pretokens = len(ins.pretokens)

        graph = nx.Graph(edges)

        dist = []
        for j in range(n_pretokens):
            sum_ = 0
            for pos in aspect_pos:
                try:
                    sum_ += nx.shortest_path_length(graph, source=j, target=pos)
                except:
                    sum_ += n_pretokens # No connection between source and target
            dist.append(sum_ / len(aspect_pos))

        all_dists[i] = dist
    
    return all_dists


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=-1)

    parser.add_argument("--process_cpc_instances", action="store_true", default=False,
        help="Whether to process CPC data from their raw data files.") 
    parser.add_argument("--process_absa_instances", action="store_true", default=False,
        help="Whether to process ABSA data from their raw data files.") 

    parser.add_argument("--generate_bert_emb", action="store_true", default=False)
    parser.add_argument("--bert_version", type=str, required=False,
        default="bert-base-uncased", help="The version of BERT. Default=`bert-base-uncased`.")

    parser.add_argument("--generate_glove_emb", action="store_true", default=False)
    parser.add_argument("--glove_dimension", type=int, required=False,
        default=100, help="The dimensions of GloVe. Default=100.")

    parser.add_argument("--generate_dep_graph", action="store_true", default=False)
    parser.add_argument("--generate_aspect_dist", action="store_true", default=False)

    args = parser.parse_args()

    # set up gpu for preprocessing
    use_gpu = False
    if args.gpu_id >= 0 and torch.cuda.device_count() > 0:
        use_gpu = True
        assert torch.cuda.device_count() > args.gpu_id
        torch.cuda.set_device("cuda:"+str(args.gpu_id))
        print("[preprocess] using GPU for processing ...")
    elif args.gpu_id < 0 and torch.cuda.device_count() > 0:
        print("[preprocess] gpu_id is set < 0, using cpu")
    else:
        print("[preprocess] gpu resource unavailable, using cpu")

    # get spacy tokenizer and bert tokenize
    nlp = en_core_web_trf.load()
    pre_tkn = nlp.tokenizer
    bert_tkn = BertTokenizer.from_pretrained(args.bert_version)

    # process or load CPC data
    if args.process_cpc_instances:
        print("[preprocess] processing cpc data ...")
        cpc_trn_data = preprocess_cpc(DATA_DIR + "data.csv", bert_tkn, pre_tkn)
        cpc_tst_data = preprocess_cpc(DATA_DIR + "held-out-data.csv", bert_tkn, pre_tkn)

        # dump data
        print("[preprocess] dumping processed cpc/absa instances to {}.".format(DATA_DIR))
        dump_pickle(DATA_DIR+"processed_cpc_train.pkl", cpc_trn_data)
        dump_pickle(DATA_DIR+"processed_cpc_test.pkl", cpc_tst_data)
    else:
        print("[preprocess] loading cpc_trn/cpc_tst data ...")
        cpc_trn_data = load_pickle(DATA_DIR+"processed_cpc_train.pkl")
        cpc_tst_data = load_pickle(DATA_DIR+"processed_cpc_test.pkl")

    # preprocess or load absa
    if args.process_absa_instances:
        print("[preprocess] processing absa data ...")
        absa_data = preprocess_absa(bert_tkn, pre_tkn)
        dump_pickle(DATA_DIR+"processed_absa.pkl", absa_data)
    else:
        print("[preprocess] loading absa data ...")
        absa_data = load_pickle(DATA_DIR+"processed_absa.pkl")

    # print cpc/absa statistics
    print("[preprocess] statistics:")
    print("\t# CPC Train:{}, CPC Test:{}, ABSA:{}".format(
        len(cpc_trn_data), len(cpc_tst_data), len(absa_data)))

    # generate bert embedding 
    if args.generate_bert_emb:
        print("[preprocess] generating BERT embedding ...")

        bert = BertModel.from_pretrained(args.bert_version, output_hidden_states=True)
        bert.eval()

        if use_gpu:
            bert = bert.cuda()

        print("\t\t CPC data ...")
        cpc_trn_bert_emb = preprocess_bert_embedding(cpc_trn_data, bert, use_gpu)
        cpc_tst_bert_emb = preprocess_bert_embedding(cpc_tst_data, bert, use_gpu)
        dump_pickle(DATA_DIR+"cpc_train_bert_emb.pkl", cpc_trn_bert_emb)
        dump_pickle(DATA_DIR+"cpc_test_bert_emb.pkl", cpc_tst_bert_emb)

        print("\t\t ABSA data ...")
        absa_bert_emb = preprocess_bert_embedding(absa_data, bert, use_gpu)
        dump_pickle(DATA_DIR+"absa_bert_emb.pkl", absa_bert_emb)

    if args.generate_glove_emb:
        print("[preprocess] generating GLOVE embedding ...")

        glove_path = "./data/glove/glove.6B.{}d.word2vec_format.txt".format(
            args.glove_dimension)
        glove = gensim.models.KeyedVectors.load_word2vec_format(glove_path)
        glove[OOV_TOK] = np.random.rand(args.glove_dimension)

        print("\t\t CPC data ...")
        cpc_trn_glove_emb = preprocess_glove_embedding(cpc_trn_data, glove, pre_tkn)
        cpc_tst_glove_emb = preprocess_glove_embedding(cpc_tst_data, glove, pre_tkn)
        dump_pickle(DATA_DIR+"cpc_train_glove_emb.pkl", cpc_trn_glove_emb)
        dump_pickle(DATA_DIR+"cpc_test_glove_emb.pkl", cpc_tst_glove_emb)
    
        print("\t\t ABSA data ...")
        absa_glove_emb = preprocess_glove_embedding(absa_data, glove, pre_tkn)
        dump_pickle(DATA_DIR+"absa_glove_emb.pkl", absa_glove_emb)

    if args.generate_dep_graph:
        print("[preprocess] generating Dependency Graph ...")
        print("\t\t CPC data ...")

        # `nlp` is the `spacy.lang.en.English` language parser class
        cpc_trn_depg = preprocess_depgraph(cpc_trn_data, nlp)
        cpc_tst_depg = preprocess_depgraph(cpc_tst_data, nlp)
        dump_pickle(DATA_DIR+"cpc_train_depgraph.pkl", cpc_trn_depg)
        dump_pickle(DATA_DIR+"cpc_test_depgraph.pkl", cpc_tst_depg)

        print("\t\t ABSA data ...")
        absa_depg = preprocess_depgraph(absa_data, nlp)
        dump_pickle(DATA_DIR+"absa_depgraph.pkl", absa_depg)
    
    else:
        print("[preprocess] loading dependency graph of cpc and absa")
        cpc_trn_depg = load_pickle(DATA_DIR+"cpc_train_depgraph.pkl")
        cpc_tst_depg = load_pickle(DATA_DIR+"cpc_test_depgraph.pkl")
        absa_depg = load_pickle(DATA_DIR+"absa_depgraph.pkl")

    if args.generate_aspect_dist:
        
        print("[preprocess] generating aspect term distance ...")
        print("\t\t CPC data ...")

        cpc_trn_aspect_distA = preprocess_aspect_dist(cpc_trn_data, cpc_trn_depg, 'A')
        cpc_tst_aspect_distA = preprocess_aspect_dist(cpc_tst_data, cpc_tst_depg, 'A')
        cpc_trn_aspect_distB = preprocess_aspect_dist(cpc_trn_data, cpc_trn_depg, 'B')
        cpc_tst_aspect_distB = preprocess_aspect_dist(cpc_tst_data, cpc_tst_depg, 'B')
        dump_pickle(DATA_DIR+"cpc_train_aspect_distA.pkl", cpc_trn_aspect_distA)
        dump_pickle(DATA_DIR+"cpc_test_aspect_distA.pkl", cpc_tst_aspect_distA)
        dump_pickle(DATA_DIR+"cpc_train_aspect_distB.pkl", cpc_trn_aspect_distB)
        dump_pickle(DATA_DIR+"cpc_test_aspect_distB.pkl", cpc_tst_aspect_distB)

        print("\t\t ABSA data ...")
        absa_aspect_dist = preprocess_aspect_dist(absa_data, absa_depg, 'A')
        dump_pickle(DATA_DIR+"absa_aspect_dist.pkl", absa_aspect_dist)