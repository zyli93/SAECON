# # Example of preprocess
# python src/preprocess.py \
#     --gpu_id 1 \
#     --process_cpc_instances \
#     --process_absa_instances \
#     --generate_bert_emb \
#     --generate_glove_emb \
#     --generate_dep_graph \
#     --generate_aspect_dist

# python src/preprocess.py \
#     --gpu_id 1 \
#     --generate_aspect_dist

# python src/preprocess.py \
#     --gpu_id 2 \
#     --generate_dep_graph \
#     --generate_aspect_dist


# python src/preprocess.py \
#     --gpu_id 1 \
#     --process_cpc_instances

# example training and validation

# NOTE:
#   1. change input_emb and emb_dim simultaneously!
#   2. feature_dim: 60, 72, 84, ...
#   3. num of optimizer
python src/train.py \
    --experimentID 00012 \
    --task train \
    --gpu_id 0 \
    --use_lr_scheduler \
    --input_emb fix \
    --emb_dim 768 \
    --feature_dim 240 \
    --lr 0.00002 \
    --absa_lr 0.0002\
    --reg_weight 0.0001 \
    --dropout 0.2 \
    --num_ep 15 \
    --batch_size 16 \
    --batch_ratio 1:1 \
    --eval_per_ep 1 \
    --eval_after_epnum 1 \
    --sgcn_dims 256 \
    --sgcn_gating \
    --sgcn_directed \
    --log_batch_num 100000 \
    --absa_log_batch_num 1000000 \
    --dom_adapt \
    --loss_weights 2 4 1 \
    --scheduler_stepsize 3
    # --data_augmentation \
    # --up_sample \
