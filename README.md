# SAECC
Sentiment Analysis Enhanced Comparative Classification
Use Saecc as the temporal name, will change it later.

## TODO list

- [ ] entityA_pos is the position of pretokens, does it matter?
- [ ] `dist` created by tokens or pretokens? ==> by tokens

## Prerequisite

To run preprocess for Saecc, we need to get ready for a few things: datasets,  Python packages, nltk corpora dependencies, pre-trained bert model, and glove pre-trained word vectors. 
### Data
#### Comparative Preference Classification (CPC)
The datasets are included in this repo. They are originally released by authors of [Categorizing Comparative Sentences](https://arxiv.org/abs/1809.06152). Here's the 
[GitHub](https://github.com/uhh-lt/comparative). Many thanks to the authors!
#### Aspect-Based Sentiment Analysis (ABSA)
The ABSA datasets are included in this repo too. They are originally released in SemEval 2014 Task 4, 2015 Task 12, and 2016 Task5. You can find them here for [2014](https://alt.qcri.org/semeval2014/task4/), [2015](https://alt.qcri.org/semeval2015/task12/), and [2016](https://alt.qcri.org/semeval2016/task5/). Many thanks to the task organizers and contributors!

We also borrowed code from this [GitHub repo](https://github.com/henrymoss/COLING2018) for processing SemEval datasets. Many thanks to the author Henry B. Moss!

### Packages
We have a compiled command file to install all packages needed. It will install a few Python packages and nltk corpora. 
```bash
bash get_ready.sh
```
Bert is also integrated in the package `transformers`. Our code will take care of downloading pretrained weights and tokenizers(many thanks to [Hugging Face](https://huggingface.co/) Team who made this possible!).

There's a special requirement for PyTorch-geometric which help GCN implementation.
On our machine, we have PyTorch version `1.8.1+cu102` and CUDA version `10.2`. Therefore, we replace `${TORCH}+${CUDA}` with `1.8.1+cu102`.
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
```


### GloVe
Please download pretrained GloVe word embeddings from [here](https://nlp.stanford.edu/projects/glove/). The dimension is open to selection. However, GloVe embedding format doesn't inherently work well with `KeyedVectors` class in `gensim`. So please first convert GloVe embeddings to word2vec format following this [link](https://radimrehurek.com/gensim/scripts/glove2word2vec.html). Transformed GloVe embedding dumps should be placed in `./data/glove/glove.6B.[dim]d.word2vec_format.txt`. The `[dim]` should be replaced by the dimension of selection.

**Cong!** You are all set.

## Preprocess

We first convert all raw data files into something that fit our implementation. 
Run
```bash
python src/preprocess.py [--options]
```
Use `-h` to see all options. For the first time, run this:
```
python src/preprocess.py --process_cpc_instances --process_absa_instances --generate_bert_emb --generate_glove_emb
```
Output files are saved in `./data/`.
