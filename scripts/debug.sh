# Example of preprocess
python src/preprocess.py \
    --gpu_id 1 \
    --process_cpc_instances \
    --process_absa_instances \
    --generate_bert_emb \
    --generate_glove_emb \
    --generate_dep_graph \
    --generate_aspect_dist

python src/preprocess.py \
    --gpu_id 1 \
    --generate_aspect_dist

python src/preprocess.py \
    --gpu_id 2 \
    --generate_dep_graph \
    --generate_aspect_dist


python src/preprocess.py \
    --gpu_id 1 \
    --process_cpc_instances

# example training and validation

# NOTE:
#   1. change input_emb and emb_dim simultaneously!
#   2. feature_dim: 60, 72, 84, ...
#   3. num of optimizer
python src/train.py \
    --experimentID 0000 \
    --task train \
    --gpu_id 0 \
    --use_lr_scheduler \
    --input_emb glove \
    --emb_dim 100 \
    --feature_dim 120 \
    --lr 0.0001 \
    --absa_lr 0.0001 \
    --reg_weight 0.00001 \
    --dropout 0.1 \
    --num_ep 3 \
    --batch_size 16 \
    --batch_ratio 1:1 \
    --data_augmentation \
    --dom_adapt \
    --eval_per_ep 1 \
    --eval_after_epnum 1 \
    --sgcn_dims 128 \
    --sgcn_gating \
    --sgcn_directed \
    --log_batch_num 20 \
    --asba_batch_num 20
