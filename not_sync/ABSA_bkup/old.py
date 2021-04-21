def preprocess_embedding(instance_features):
    bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    bert.eval()

    # do not run the back propagation.
    with torch.no_grad():

        for ins in tqdm(instance_features):
            # convert python lists to torch tensors
            tokens_tensor = torch.tensor([ins.get_token_ids()])
            mask_tensors = torch.tensor([ins.get_token_mask()])
            output = bert(tokens_tensor, mask_tensors)

            embedding = output.last_hidden_state

            # squeeze the tensor to remove the batch
            embedding = torch.squeeze(embedding, dim=0)  # (token_len, dim)

        tokenized_sentence = instance_feature.get_tokens()

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