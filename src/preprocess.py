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
from utils import InstanceFeatures, Embeddings
from transformers import BertTokenizer, BertModel


# Return a list of InstanceFeatures. One InstanceFeature for each sentence.
def preprocess_bert_tokenizer(file_path):

    # return values
    all_instance_features = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            sentence = row['sentence']
            label = row['most_frequent_label']

            tokenizer_output = tokenizer(sentence)

            index_tokens = tokenizer_output['input_ids']
            mask = tokenizer_output['attention_mask']

            # len(tokens) = index_tokens - 2 because [CLS] and [SEP] are skipped
            tokens = tokenizer.convert_ids_to_tokens(index_tokens, skip_special_tokens=True)
            sentence_from_tokens = tokenizer.convert_tokens_to_string(tokens)


            token_to_ori_map = []

            for i in range(0, len(tokens)):
                if i == 0:
                    token_to_ori_map.append(0)
                else:
                    if len(tokens[i]) > 2 and tokens[i][0:2] == "##":
                        token_to_ori_map.append(token_to_ori_map[-1])
                    else:
                        token_to_ori_map.append(token_to_ori_map[-1] + 1)

            label_id = -1

            if label == "BETTER":
                label_id = 0
            elif label == "WORSE":
                label_id = 1
            else:
                label_id = 2

            all_instance_features.append(InstanceFeatures(tokens, index_tokens, mask, label,
                                                          label_id, token_to_ori_map, sentence_from_tokens, None))

    return all_instance_features


def preprocess_embedding(instance_feature):

    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()

    # do not run the back propagation.
    with torch.no_grad():

        # convert python lists to torch tensors
        tokens_tensor = torch.tensor([instance_feature.get_token_ids()])
        segments_tensors = torch.tensor([instance_feature.get_token_mask()])

        output = model(tokens_tensor, segments_tensors)

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



all_instance = preprocess_bert_tokenizer("../data/data.csv")

print(len(all_instance))
print(all_instance[0].get_token_to_orig_map())

embedding = preprocess_embedding(all_instance[0])

print(embedding.get_embedding().size())
print(embedding.get_embedding_without_word_piece().size())



