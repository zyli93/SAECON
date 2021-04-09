from ABSA.saecc_train import SAECC_ABSA
from utils import dynamic_padding
from model import AbsaPipeline
import pickle
import math


epoch = 4

all_emb = all_ins = []

with open("./ABSA/datasets/absa/absa_bert_emb_cpu.pkl", "rb") as fin:
    all_emb = pickle.load(fin)

with open("./ABSA/datasets/absa/processed_absa.pkl", "rb") as fin:
    all_ins = pickle.load(fin)

instance = SAECC_ABSA(16)
pipeline = AbsaPipeline(16)

print(type(all_emb))
print(type(all_ins))

for each_epoch in range(0, epoch):
    pipeline.reset_stats()

    for i in range(math.floor(len(all_emb) / 16)):
        batch_data = []

        batch_emb = []
        batch_ins = []
        for j in range(0, 16):
            emb = all_emb[i * 16 + j]
            ins = all_ins[i * 16 + j]

            batch_emb.append(emb)
            batch_ins.append(ins)

        batch_res = pipeline({'embedding': batch_emb, 'instance_feature': batch_ins})

        print(batch_res)


        # for j in range(0, 16):
        #     padded_emb, _ = dynamic_padding([all_emb[i * 16 + j]], 80)
        #     each_sent = []
        #     each_sent.append(padded_emb[0])
        #     each_sent.append(all_ins[i * 16 + j])
        #
        #     batch_data.append(each_sent)
        #
        # # batch_res = instance.run_batch(batch_data)
        #
        # batch_res = pipeline(batch_data)
        #
        # print(batch_res)
