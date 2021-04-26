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
    --gpu_id 1 \
    --process_cpc_instances

# example training and validation
python scr/train.py \
    --experimentID 0000 \
    --task train \
    --gpu_id 1 \
    --use_lr_scheduler \
    --input_emb ft \
    --emb_dim 768 \
    --feature_dim 128 \
    --lr 0.0001 \
    --reg_weight 0.00001 \
    --dropout 0.1 \
    --num_ep 3 \
    --batch_size 16 \
    --bath_ratio 1:1 \
    --data_augmentation \
    --dom_adapt \
    --eval_per_ep 1 \
    --eval_after_ep 1
