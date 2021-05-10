"""
    Entrance for Training SAECC model

    Authors:
        Anon <anon@anon.anon>
    Date created: March 11, 2020
    Python version: 3.6.0+
"""

import os
import pdb
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import SaeccModel
# from model_ablation import SaeccModel
from dataloader import DataLoader
from dataloader import CPC, ABSA, BATCH_TENSORS
from utils import make_dir, print_args, get_time
from utils import LOG_DIR, DATA_DIR, CKPT_DIR
from utils import LABELS
from utils import ID2LABEL, ID2LABEL_ABSA
from sklearn.metrics import f1_score

import wandb

def get_optimizer_and_scheduler(args, task, model):
    # for CPC model, update all parameters as they are all involved.
    # for ABSA model, update only absa pipeline.
    params = model.parameters() if task == CPC \
        else model.absa_pipeline.parameters()
    lr = args.lr if task == CPC else args.absa_lr

    # create optimizer
    optimizer = optim.Adam(
        params, lr=lr, weight_decay=args.reg_weight)

    # create scheduler, if not scheduler, return None
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.scheduler_stepsize, 
            gamma=args.scheduler_gamma)
    else:
        scheduler = None
    
    return optimizer, scheduler


def compose_msg(task, ep, batch_count, ttl_iter_num, loss, 
    ttl_cpc_loss, cpc_batch_count, ttl_absa_loss, absa_batch_count):
    msg = f"[{task}] ep:[{ep}] iter:[{batch_count}"
    msg += f"/{ttl_iter_num}] "
    msg += f"loss:[{loss:.6f}] "
    
    msg += "accumulate loss: CPC-[{}]; ABSA-[{}]".format(
        ttl_cpc_loss / (cpc_batch_count + 1E-6),
        ttl_absa_loss / (absa_batch_count + 1E-6)
    )
    return msg


def move_batch_to_device(batch, device):
    """get all entries of tensors in the batch by `BATCH_TENSORS`
    and move them to GPU"""
    for field in BATCH_TENSORS:
        batch[field] = batch[field].to(device)
    return batch


def compute_metrics(pred_list, gt_list):
    """
    computing the performance using two lists of vectors
    Args:
        pred_list - list of tensor on device for predictions
        gt_list - lits of tensor on device for ground truths
    Return:
        metric_dict = 
            {
                0: f1(cls 0),
                1: f1(cls 1),
                2: f1(cls 2),
                "micro": f1(micro)
            }
    """
    pred_list = [x.to("cpu").detach().numpy() for x in pred_list]
    gt_list = [x.to("cpu").detach().numpy() for x in gt_list]
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(gt_list)
    each_class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None)
    metric_dict = dict(zip(LABELS, each_class_f1))
    metric_dict["micro"] = f1_score(y_true, y_pred, average="micro")
    return metric_dict


def compose_metric_perf_msg(metric_dict):
    perf_msg = " ".join([f"{ID2LABEL[i]}-{metric_dict[i]:.6f};" for i in LABELS])
    perf_msg += f" micro-{metric_dict['micro']}"
    return perf_msg
    

def gen_domain_target(batch):
    """Generate domain target. 
        For CPC, generate (batch_size) zero vector
        For ABSA, generate (2*batch_size) one vector
    """
    batch_size = len(batch['instances'])
    if batch['task'] == CPC:
        return torch.from_numpy(np.zeros(2*batch_size))
    else:
        return torch.from_numpy(np.ones(batch_size))


def train(args, device, model, dataloader):

    logging.info(f"[info] start training ID:[{args.experimentID}]")
    print(f"[info] start training ID:[{args.experimentID}]")

    # create optimizer(s) and schedulers
    # configs of optimizers and schedulers depend on `use_single_optimizer`
    # and `use_lr_scheduler`.
    cpc_optim, cpc_sched = get_optimizer_and_scheduler(args, CPC, model)
    if not args.use_single_optimizer:
        absa_optim, absa_sched = get_optimizer_and_scheduler(args, ABSA, model)

    # create loss function
    loss_weights = [float(x) for x in args.loss_weights]
    print("Using loss weights ", loss_weights)
    criterion_cpc = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights).to(device))
    criterion_absa = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss()
    if args.dom_adapt:
        dom_criterion = nn.BCEWithLogitsLoss()

    total_iter_counter = 0

    for ep in range(args.num_ep):
        trn_iter = dataloader.get_batch_train()
        total_iter_num_per_epoch = dataloader.trn_batch_num
        model.train()

        total_cpc_loss, total_absa_loss = 0., 0.
        cpc_batch_count, absa_batch_count = 0, 0
        predictions, groundtruths = [], []  # used to compute training performance

        print(f"{get_time()} [Time] Starting Epoch {ep}")
        logging.info(f"[Time] Starting Epoch {ep}")

        for bid, batch in enumerate(trn_iter):

            task = batch['task']

            # change it to use gpu
            batch = move_batch_to_device(batch, device)
            
            # get prediction and target
            model_out = model(batch)
            pred_logits = model_out['prediction']  # logits
            target = torch.tensor(
                [x.get_label_id() for x in batch['instances']]).to(device)

            # choose optimizer and zero grad
            if not args.use_single_optimizer and task == ABSA:
                optim = absa_optim
            else:
                optim = cpc_optim
            optim.zero_grad()

            # compute loss, compute derivation, optimize
            criterion = criterion_cpc if task == CPC else criterion_absa
            loss = task_loss = criterion(pred_logits, target)
            
            # TODO: check optimizer
            dom_loss = None
            if args.dom_adapt:
                dom_pred = model_out['domain_logit'].squeeze()
                dom_target = gen_domain_target(batch).to(device)
                dom_loss = dom_criterion(dom_pred, dom_target)
                loss = task_loss + dom_loss
                # TODO: do we need a hyperparam here
            loss.backward()
            optim.step()

            # accumulate loss
            if task == CPC:
                cpc_batch_count += 1
                total_cpc_loss += loss.item()
                # collect prediction results and ground truths
                # pred: softmax+argmax on dim 1
                pred = torch.argmax(torch.softmax(pred_logits, 1), 1)
                predictions.append(pred)
                groundtruths.append(target)

                if not cpc_batch_count % args.log_batch_num:
                    msg = compose_msg(CPC, ep, cpc_batch_count, 
                        total_iter_num_per_epoch, loss.item(), 
                        total_cpc_loss, cpc_batch_count,
                        total_absa_loss, absa_batch_count)

                    if args.use_wandb:
                        wandb.log({
                            "cpc_loss": task_loss.item(),
                            "dom_loss": dom_loss.item() if dom_loss else 0,
                            "accumulated_cpc_loss": total_cpc_loss / (cpc_batch_count + 1E-6),
                            "accumulated_absa_loss": total_absa_loss / (absa_batch_count + 1E-6),
                        })

                    logging.info("[Perf][Iter] " + msg)
                    print(f"{get_time()} [Perf][Iter] {msg}")
            else:
                absa_batch_count += 1
                total_absa_loss += loss.item()
                if not absa_batch_count % args.absa_log_batch_num:
                    msg = compose_msg(ABSA, ep, cpc_batch_count, 
                        total_iter_num_per_epoch, loss.item(), 
                        total_cpc_loss, cpc_batch_count,
                        total_absa_loss, absa_batch_count)
                    
                    if args.use_wandb:
                        wandb.log({
                            "absa_loss": task_loss.item(),
                            "dom_loss": dom_loss.item() if dom_loss else 0,
                            "accumulated_cpc_loss": total_cpc_loss / (cpc_batch_count + 1E-6),
                            "accumulated_absa_loss": total_absa_loss / (absa_batch_count + 1E-6)
                        })
            
                    logging.info("[Perf][Iter] " + msg)
                    print(f"{get_time()} [Perf][Iter] {msg}")

            
        # log training loss after each epoch
        msg = compose_msg(CPC, ep, 0, 0, 0, total_cpc_loss, cpc_batch_count,
            total_absa_loss, absa_batch_count)
        logging.info("[Perf][Epoch] " + msg)
        print(f"{get_time()} [Perf][Epoch] {msg}")

        # compute and log training performance after each epoch
        # TODO: uncomment this!
        # metric_dict = compute_metrics(predictions, groundtruths)
        # perf_msg = compose_metric_perf_msg(metric_dict)
        # if args.use_wandb:
        #     wandb.log({'train F1-' + str(k): v for k, v in metric_dict.items()})
        # logging.info(f"[Perf][Train][CPC][Epoch]{ep} " + perf_msg)
        # print(f"{get_time()} [Perf][Train][CPC][Epoch]{ep} " + perf_msg)

        # averaging loss for epoch
        if args.use_lr_scheduler:
            cpc_sched.step()
            if absa_sched:
                absa_sched.step()

        # run validation
        if not ep % args.eval_per_ep and ep >= args.eval_after_epnum - 1:
            # TODO: uncomment this!
            metric_dict, perf_msg = evaluate(model, for_test=False,
                data_iter=dataloader.get_batch_testval(False), 
                restore_model_path=None, device=device)
            if args.use_wandb:
                wandb.log({'eval F1-' + str(k): v for k, v in metric_dict.items()})
            logging.info(f"[Perf-CPC][val][epoch]{ep} {perf_msg}")
            print(f"{get_time()} [Perf-CPC][val][epoch]{ep} {perf_msg}")

            _, perf_msg = evaluate(model, 
                data_iter=dataloader.get_batch_testval(for_test=True),
                restore_model_path=None, 
                device=device, for_test=True)
            logging.info(f"[Perf-CPC][Test] {perf_msg}")
            print(f"{get_time()} [Perf-CPC][Test] {perf_msg}")

        # save model
        if args.save_model:
            model_name = "model_expID{}_ep{}".format(args.experimentID, ep)
            torch.save(model.state_dict(), args.save_model_path+model_name)
            logging.info(f"[save] saving model: {model_name}")
            print(f"{get_time()} [save] saving model: {model_name}")


def evaluate(model, data_iter, restore_model_path, device, for_test):
    """
    Run validation or test, return a performance msg in metrics
    Args:
        model - the model
        data_iter - the data iterator for test/val data
        restore_model_path - the path to restore the current model
        for_test - True for testing, False for validation
    Return:
        perf_msg - performance message composed by `compose_metric_perf_msg`
    """
    model.eval()
    predictions, groundtruths = [], []

    task = "Test" if for_test else "Val"

    if restore_model_path:
        logging.info(f"[Eval] loading model from {restore_model_path}")
        print(f"[Eval] loading model from {restore_model_path}")
        model.load_state_dict(torch.load(restore_model_path))

<<<<<<< HEAD
    all_entA = all_entB = []
=======
    entityA = entityB = []
>>>>>>> b62679a... debug
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            batch = move_batch_to_device(batch, device)
            eval_pred = model(batch)
            eval_logits = eval_pred['prediction']
<<<<<<< HEAD
            print(len(eval_pred))
            # eval_entA = eval_pred['entityA']
            # eval_entB = eval_pred['entityB']
=======
            if for_test:
                entityA.append(eval_pred['entityA'])
                entityB.append(eval_pred['entityB'])
>>>>>>> b62679a... debug
            eval_pred = torch.argmax(torch.softmax(eval_logits, 1), 1)
            eval_groundtruth = torch.tensor(
                [x.get_label_id() for x in batch['instances']])
            # eval_pred = eval_pred.cpu()
            # eval_groundtruth = eval_groundtruth.cpu()
            predictions.append(eval_pred)
            groundtruths.append(eval_groundtruth)

            # all_entA.append(eval_entA)
            # all_entB.append(eval_entB)

        # compute metric performance
        metric_dict = compute_metrics(predictions, groundtruths)
        # compose a message for performance
        perf_msg = compose_metric_perf_msg(metric_dict)
    
    return metric_dict, perf_msg, entityA, entityB

def setup_wandb(args):
    wandb.init(project='saecc', entity='louixp')
    wandb.config.update(args)
    args.experimentID = wandb.run.name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # environment config
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--experimentID", type=str, help="The ID of the experiments")
    parser.add_argument("--task", type=str, required=True, help="train/test?")
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--shuffle", action="store_true", default=False,
        help="Whether to shuffle data before a new epoch")
    parser.add_argument("--up_sample", action="store_true", default=False,
        help="Upsample dataset inside dataloader")

    # scheduler
    parser.add_argument("--use_lr_scheduler", action="store_true", default=False,
        help="Use this flag to turn on learning rating scheduler in model")
    parser.add_argument("--scheduler_stepsize", type=int, default=5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.8)

    # input embedding
    parser.add_argument("--input_emb", type=str, default="bert_fixed", 
        help="Select from `fix`, `ft`, and `glove`")
    parser.add_argument("--bert_version", type=str, default="bert-base-uncased")
    parser.add_argument("--emb_dim", type=int, 
        help="Embedding dimension of Bert or Glove embedding")
    parser.add_argument("--glove_dim", type=int, default=100)
    parser.add_argument("--feature_dim", type=int, default=100) 

    # training config
    parser.add_argument("--loss_weights", nargs="+", type=float, default=[1.,1.,1.],
        help="weight of loss")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--reg_weight", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_ep", type=int, default=10)
    parser.add_argument("--use_single_optimizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_ratio", type=str, default="1:1", 
        help="Ratio of batch numbers for CPC and ABSA. Default: 1:1.")
    parser.add_argument("--data_augmentation", action="store_true", default=False,
        help="Whether to conduct data augmentation for training set. Default=False.")

    # cpc model config
    parser.add_argument("--sgcn_dims", nargs='+', type=int)
    parser.add_argument("--sgcn_gating", action="store_true", default=False)
    parser.add_argument("--sgcn_directed", action="store_true", default=False)
    
    # absa model config
    parser.add_argument("--absa_lr", type=float, default=0.005)
    parser.add_argument("--absa_max_seq_len", type=str, default=80)
    parser.add_argument("--absa_dropout", type=float, default=0.2)
    parser.add_argument("--absa_local_context_focus", type=str, default="cdw",
        help="Local context focus option: can be `cdm` or `cdw`")
    parser.add_argument("--absa_syntactic_relative_distance", type=int, default=4,
        help="Syntactic relative distance")
    
    parser.add_argument("--activation", type=str, default="relu",
        help="Activation function to use in our model")
    parser.add_argument("--dom_adapt", action="store_true", default=False,
        help="Whether to do domain adaptation")

    # logging 
    parser.add_argument("--log_batch_num", type=int, default=100)
    parser.add_argument("--absa_log_batch_num", type=int, default=100)
    
    # validation
    parser.add_argument("--eval_per_ep", type=int, default="100",
        help="Run validation every this number of epochs. Default=100 (No val)")
    parser.add_argument("--eval_after_epnum", type=int, default="100",
        help="Start to run validation after this num of epochs. Default=100 (No val)")
    
    # model saving
    parser.add_argument("--save_model", action="store_true", default=False, 
        help="Whether to turn on model saving.")
    parser.add_argument("--save_per_ep", type=int, default=2, help="Save model per x epochs.")
    parser.add_argument("--save_after_epnum", type=int, default=1000,
        help="Number of iterations per model saving." + 
             "Only in effect when `save_model` is turned on.")
    parser.add_argument("--save_model_path", type=str, default="./ckpt/",
        help="Path to directory to save models")
    
    # model loading
    parser.add_argument("--load_model", action="store_true", default=False,
        help="Whether to resume training from an existing ckpt")
    parser.add_argument("--load_model_path", type=str, help="Path to load existing model")

    args = parser.parse_args()

    if args.use_wandb:
        setup_wandb(args)

    # hacking feature_dim 
    # override gpu_id 
    # https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/5
    def get_freer_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available)

    args.gpu_id = get_freer_gpu()

    # make directories
    make_dir(LOG_DIR)
    make_dir(CKPT_DIR)

    print("="* 20 + "\n  ExperimentID " + args.experimentID + "\n" + "="*20)
    print_args(args)

    # config logging
    logging.basicConfig(filename=LOG_DIR+'{}.log'.format(args.experimentID), 
        filemode='w+', level=logging.DEBUG, 
        format='[%(asctime)s][%(levelname)s][%(filename)s] %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S')
    
    # Setup GPU device
    if torch.cuda.device_count() > 0:
        assert torch.cuda.device_count() > args.gpu_id
        device = f"cuda:{args.gpu_id}"
        torch.cuda.set_device("cuda:"+str(args.gpu_id))
        msg = "[cuda] with {} gpus, using cuda:{}".format(
            torch.cuda.device_count(), args.gpu_id)
    else:
        device = "cpu"
        msg = "[cuda] no gpus, using cpu"


    logging.info(msg)
    print("{} {}".format(get_time(), msg))

    # set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    dataloader = DataLoader(args)

    # move model to cuda device
    model = SaeccModel(args, device)
    model = model.to(device)

    if args.task == "train":
        train(args, device, model, dataloader)
    elif args.task == "test":
<<<<<<< HEAD
        _, perf_msg, all_entA, all_entB = evaluate(model,
=======
        _, perf_msg, entityA, entityB = evaluate(model, 
>>>>>>> b62679a... debug
            data_iter=dataloader.get_batch_testval(for_test=True),
            restore_model_path=args.load_model_path, 
            device=device, for_test=True)
        logging.info(f"[Perf-CPC][Test] {perf_msg}")
        print(f"{get_time()} [Perf-CPC][Test] {perf_msg}")
<<<<<<< HEAD
        print(all_entA[0])
        print(all_entB[0])
=======
        print("Shape:")
        print(str(len(entityA)) + " " + str(len(entityB)))
        print(str(len(entityA[0])) + " " + str(len(entityB[0])))
        print("Sample:")
        print(str(entityA[0]) + " " + str(entityB[0]))
        print("0 1")
        print(str(entityA[0][1]) + " " + str(entityB[0][1]))
>>>>>>> b62679a... debug
    else:
        raise ValueError("args.task can only be train or test")
