from itertools import product
import os

exp_id_base = "grid_ratio_{}_{}"
end = 5
test_range = product(range(1,end+1), range(1,end+1))
original_setting = """python src/train.py \
    --experimentID {}\
    --task train \
    --gpu_id 0 \
    --use_lr_scheduler \
    --input_emb fix \
    --emb_dim 768 \
    --feature_dim 72 \
    --lr 0.0001 \
    --absa_lr 0.0002\
    --reg_weight 0.00001 \
    --dropout 0.1 \
    --num_ep 10 \
    --batch_size 16 \
    --batch_ratio {} \
    --dom_adapt \
    --eval_per_ep 1 \
    --eval_after_epnum 1 \
    --sgcn_dims 256 \
    --sgcn_gating \
    --sgcn_directed \
    --log_batch_num 1 \
    --absa_log_batch_num 1"""

for cpc_batch, absa_batch in test_range:
    ratio = f"{cpc_batch}:{absa_batch}"
    exp_id = exp_id_base.format(cpc_batch, absa_batch)
    cmd = original_setting.format(
        exp_id,
        ratio)
    
    print(cmd)
    os.system(cmd)

