import sys
import logging
import torch
from torch import nn, optim
from utils import get_time

from model import EDGAT
from dataloader import DataLoader
from train import move_batch_to_device, compose_msg, compute_metrics, compose_metric_perf_msg

class Args:
    num_ep = 10
    batch_size = 32
    batch_ratio = "1:0"
    input_emb = "fix"
    data_augmentation = False
    up_sample = False

dataloader = DataLoader(Args)
device = "cuda:0"
model = EDGAT(768, 8, device).to(device)
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(
    model.parameters(), 
    lr=5e-4
)

# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(pytorch_total_params)

# for p in model.parameters():
#     if p.requires_grad:
#         print(p.numel())
# sys.exit()

for ep in range(Args.num_ep):
    trn_iter = dataloader.get_batch_train()
    total_iter_num_per_epoch = dataloader.trn_batch_num
    model.train()

    total_cpc_loss = 0.
    cpc_batch_count = 0
    predictions, groundtruths = [], []  # used to compute training performance

    print(f"[{get_time()}] Starting Epoch {ep}")

    for bid, batch in enumerate(trn_iter):

        task = batch['task']

        # change it to use gpu
        batch = move_batch_to_device(batch, device)
        
        # get prediction and target
        pred_logits = model(batch)
        target = torch.tensor(
            [x.get_label_id() for x in batch['instances']]).to(device)

        optim.zero_grad()

        # compute loss, compute derivation, optimize
        loss = criterion(pred_logits, target)
        
        loss.backward()
        optim.step()

        # accumulate loss
        cpc_batch_count += 1
        total_cpc_loss += loss.item()
        # collect prediction results and ground truths
        # pred: softmax+argmax on dim 1
        pred = torch.argmax(torch.softmax(pred_logits, 1), 1)
        predictions.append(pred)
        groundtruths.append(target)

        msg = compose_msg(task, ep, cpc_batch_count, 
            total_iter_num_per_epoch, loss.item(), 
            total_cpc_loss, cpc_batch_count,
            0, 0)

        logging.info("[Perf][Iter] " + msg)
        
    # log training loss after each epoch
    msg = compose_msg("cpc", ep, 0, 0, 0, total_cpc_loss, cpc_batch_count, 0, 0)
    print(get_time() + "[Perf][Epoch] " + msg)

    # compute and log training performance after each epoch
    metric_dict = compute_metrics(predictions, groundtruths)
    perf_msg = compose_metric_perf_msg(metric_dict)
    print(f"{get_time()}[Perf][Train][CPC][Epoch]{ep} " + perf_msg)

    # run validation
    model.eval()
    predictions, groundtruths = [], []

    task = "Test"

    # with torch.no_grad():
    #     for i, batch in enumerate(dataloader.get_batch_testval(for_test=True)):
    #         batch = move_batch_to_device(batch, device)
    #         eval_logits = model(batch)
    #         eval_pred = torch.argmax(torch.softmax(eval_logits, 1), 1)
    #         eval_groundtruth = torch.tensor(
    #             [x.get_label_id() for x in batch['instances']])
    #         predictions.append(eval_pred)
    #         groundtruths.append(eval_groundtruth)

    #     # compute metric performance
    #     metric_dict = compute_metrics(predictions, groundtruths)
    #     # compose a message for performance
    #     perf_msg = compose_metric_perf_msg(metric_dict)
    
    # print(f"[Perf-CPC][val][epoch]{ep} {perf_msg}")