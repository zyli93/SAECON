python src/train.py \
    --experimentID ExampleID\
    --task train \
    --gpu_id 0 \
    --use_lr_scheduler \
    --input_emb fix \
    --emb_dim 768 \
    --feature_dim 240 \
    --lr 0.0005 \
    --absa_lr 0.0002\
    --reg_weight 0.0001 \
    --dropout 0.2 \
    --num_ep 10 \
    --batch_size 16 \
    --batch_ratio 1:1 \
    --eval_per_ep 1 \
    --eval_after_epnum 1 \
    --sgcn_dims 256 \
    --sgcn_gating \
    --sgcn_directed \
    --log_batch_num 100000 \
    --absa_log_batch_num 1000000 \
    --loss_weights 2 4 1 \
    --scheduler_stepsize 3 \
    --save_model \
    --save_per_ep 9 \
    --save_after_epnum 8 \
    --dom_adapt
    # --data_augmentation \
    # --up_sample \
