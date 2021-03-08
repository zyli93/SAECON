"""
    Preprocess input CPC file and output InstanceFeatures

    Authors:
        Zihan Liu <leoliu00529@gmail.com>
        Zeyu Li <zyli@cs.ucla.edu>
    Date created: March 7, 2021
    Python version: 3.6.0+
"""

import csv
import torch
import argparse
from utils import InstanceFeatures, Embeddings
from transformers import BertTokenizer, BertModel

from utils import dump_pickle

DATA_DIR = "./data/"


# Return a list of InstanceFeatures. One InstanceFeature for each sentence.
def preprocess_bert_tokenizer(file_path):

    # return values
    all_instance_features = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    max_ = 0

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            sentence = row['sentence']
            label = row['most_frequent_label']

            tokenizer_output = tokenizer(sentence)

            index_tokens = tokenizer_output['input_ids']
            mask = tokenizer_output['attention_mask']

            # [Deprecated] len(tokens) = index_tokens - 2 because [CLS] and [SEP] are skipped
            # keep the special tokens inside the `tokens`
            tokens = tokenizer.convert_ids_to_tokens(index_tokens)
            sentence_from_tokens = tokenizer.convert_tokens_to_string(tokens)

            if max_ != 10:
                max_ += 1
            else:
                break

            print(sentence_from_tokens)

            token_to_orig_map_list = []
            token_to_orig_map = {}

            # ranging from 1 to len(tokens)-1
            token_indices = list(range(1, len(tokens) - 1))
            for i in token_indices:
                if i == 1:
                    token_to_orig_map_list.append(0)
                else:
                    if len(tokens[i]) > 2 and tokens[i][0:2] == "##":
                        token_to_orig_map_list.append(token_to_orig_map_list[-1])
                    else:
                        token_to_orig_map_list.append(token_to_orig_map_list[-1] + 1)
            
            assert len(token_indices) == len(token_to_orig_map_list), "Unequal lengths!"
            token_to_orig_map = dict(zip(token_indices, token_to_orig_map_list))

            label_id = -1

            if label == "BETTER":
                label_id = 0
            elif label == "WORSE":
                label_id = 1
            else:
                label_id = 2

            # TODO: 
            #   1. add dep parsing graph
            #   2. add glove embedding indices
            all_instance_features.append(
                InstanceFeatures(tokens, index_tokens, mask, label, 
                                 label_id, token_to_orig_map, sentence_from_tokens, 
                                 None))

    return all_instance_features


def preprocess_embedding(bert, instance_feature):


    # do not run the back propagation.
    with torch.no_grad():

        # convert python lists to torch tensors
        tokens_tensor = torch.tensor([instance_feature.get_token_ids()])
        mask_tensors = torch.tensor([instance_feature.get_token_mask()])

        output = bert(tokens_tensor, mask_tensors)

        embedding = output.last_hidden_state

        # squeeze the tensor to remove the batch
        embedding = torch.squeeze(embedding, dim=0)

        tokenized_sentence = ["[CLS]"] + instance_feature.get_tokens() + ["[SEP]"]

        token_no_wordpiece = []
        embedding_no_wordpiece = []

        # Loop through all tokens. If a word piece is found, sum the embeddings of all related word pieces and
        # calculate the average. The average valued tensor is then used as the embedding of the word.
        # Ex: embedding of the word "embeddings" is the avergae of the embeddings of "em", "##bed", "##ding", "##s".

        for i in range(0, len(tokenized_sentence)):

            each_token = tokenized_sentence[i]

            if len(each_token) > 2 and each_token[0:2] == "##":

                if len(tokenized_sentence[i - 1]) > 2 and tokenized_sentence[i - 1][0:2] == "##":
                    continue

                num_sequence = 2
                sum_embedding = embedding[i - 1].add(embedding[i])
                sum_token = each_token[2:]
                for j in range(i + 1, len(tokenized_sentence)):

                    next_token = tokenized_sentence[j]

                    if len(next_token) > 2 and next_token[0:2] == "##":
                        num_sequence += 1
                        sum_embedding = sum_embedding.add(embedding[j])
                        sum_token += next_token[2:]
                    else:
                        break

                token_no_wordpiece[-1] = token_no_wordpiece[-1] + sum_token
                embedding_no_wordpiece[-1] = torch.div(sum_embedding, num_sequence)
            else:
                token_no_wordpiece.append(tokenized_sentence[i])
                embedding_no_wordpiece.append(embedding[i])

        return Embeddings(embedding, torch.stack(embedding_no_wordpiece))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--generate_bert_emb", action="store_true", default=False)
    parser.add_argument("--bert_version", type=str, required=False,
        default="bert-base-uncased", help="The version of BERT.")
    args = parser.parse_args()

    all_instance = preprocess_bert_tokenizer(DATA_DIR + "data.csv")

    print("[preprocess] dumping processed instances to {}.".format(args.output_path))
    dump_pickle(args.output_path, all_instance)

    print(len(all_instance))
    print(all_instance[0].get_token_to_orig_map())

    if args.generate_bert_emb:
        
        bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        bert.eval()
        print("[preprocess] generating BERT embedding ...")
        embedding = preprocess_embedding(bert, all_instance[0])

        print(embedding.get_embedding().size())
        print(embedding.get_embedding_without_word_piece().size())



