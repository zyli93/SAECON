class Tokenizer4Pretrain:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        # sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        sequence = self.tokenizer(text)['input_ids']
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_or_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    # Group distance to aspect of an original word to its corresponding subword token
    def retok_with_dist(self, text, dep_dist)
        """
        convert spacy tokenization to bert tokenization
        assign distance of each spacy-token to resultant bert-token
        """
        distances = []
        for word, dist in zip(text, dep_dist):
            tokens = self.tokenizer.tokenize(word)
            distances += [dist] * len(tokens)

        if len(distances) == 0:
            distances = [0]
        token_dist = pad_or_truncate(distances, self.max_seq_len)
        return token_dist


class ABSADataset(Dataset):
    def __init__(self, inputs, tokenizer):

        all_data = []

        # for using bert embeddings as input


        for each_sent in inputs:

            bert_embedding = each_sent[0]
            instance_feature = each_sent[1]
            entity_A, _ = instance_feature.get_entities()
            # polarity = instance_feature.get_label_id()

            text_raw_bert_indices = tokenizer.text_to_sequence(tokenizer.cls_token + ' ' + instance_feature.sentence
                                                           + ' ' + tokenizer.sep_token)
            aspect_bert_indices = tokenizer.text_to_sequence(tokenizer.cls_token + ' ' + entity_A
                                                             + ' ' + tokenizer.sep_token)
            raw_tokens, dist = calculate_dep_dist(instance_feature.sentence, entity_A)
            raw_tokens.insert(0, tokenizer.cls_token)
            dist.insert(0, 0)
            raw_tokens.append(tokenizer.sep_token)
            dist.append(0)

            _, dep_distance_to_aspect = tokenizer.tokenize(raw_tokens, dist)

            data = {
                'bert_embedding': bert_embedding,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'dep_distance_to_aspect': dep_distance_to_aspect,
                # 'polarity': polarity
            }

            all_data.append(data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)