from itertools import product
import os

exp_id_base = "DG_{}_{}"
gating_arg = " --sgcn_gating "
directed_arg = " --sgcn_directed "
test_range = product([True, False], [True, False])
original_setting = """python src/train.py \
    --experimentID {}\
    --task train \
    --gpu_id 1 \
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
    --batch_ratio 1:1 \
    --dom_adapt \
    --eval_per_ep 1 \
    --eval_after_epnum 1 \
    --sgcn_dims 256 {} {} \
    --log_batch_num 1 \
    --absa_log_batch_num 1"""

for do_g, do_d in test_range:
    print(do_g, do_d)
    gating = gating_arg if do_g else ""
    directed = directed_arg if do_d else ""

    exp_id = exp_id_base.format("D" if do_d else "UnD", "G" if do_g else "UnG")
    cmd = original_setting.format(exp_id, gating, directed)
    
    print(cmd)
    os.system(cmd)
