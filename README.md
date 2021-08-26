# SAECON
Sentiment Analysis Enhanced COmparative Network

## Reference

The paper associated with this GitHub repo has been accepted by EMNLP'21 Findings. Please cite our paper via the following BibTex.
```
@inproceedings{saecon,
  author    = {Zeyu Li and
               Yilong Qin and
               Zihan Liu and
               Wei Wang},
  title     = {Powering Comparative Classification with Sentiment Analysis via Domain Adaptive Knowledge Transfer},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural
               Language Processing: Findings, {EMNLP} 2021, 7-11 November 2021, 
               Online and in the Barceló Bávaro Convention Centre, Punta Cana, Dominican Republic},
  series    = {Findings of {ACL}},
  volume    = {{EMNLP} 2021},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
}
```


## Prerequisite

To run preprocess for SAECON, we need to get ready for a few things: datasets,  Python packages, nltk corpora dependencies, pre-trained bert model, and glove pre-trained word vectors. 
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
pip install torch-geometric
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
python src/preprocess.py \
    --gpu_id 1 \
    --process_cpc_instances \
    --process_absa_instances \
    --generate_bert_emb \
    --generate_glove_emb \
    --generate_dep_graph \
    --generate_aspect_dist
```
Output files are saved in `./data/`.

## Training
To run the training pipelines, use
```
python src/train.py [--options]
```
Use `-h` to see all options. Example usage:
```
python src/train.py \
    --task train \
    --gpu_id 0 \
    --use_lr_scheduler \
    --input_emb fix \
    --emb_dim 768 \
    --feature_dim 120 \
    --lr 0.0005 \
    --absa_lr 0.002 \
    --reg_weight 0.00001 \
    --dropout 0.1 \
    --num_ep 50 \
    --batch_size 16 \
    --batch_ratio 1:1 \
    --eval_per_ep 1 \
    --eval_after_epnum 1 \
    --sgcn_dims 256 \
    --sgcn_gating \
    --sgcn_directed \
    --log_batch_num 1 \
    --absa_log_batch_num 1 
```

## Testing

