"""
    Entrance for Training SAECC model

    Authors:
        Anon <anon@anon.anon>
    Date created: March 11, 2020
    Python version: 3.6.0+
"""

import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import SaeccModel
from dataloader import DataLoader
from dataloader import CPC, ABSA, BATCH_TENSORS
from utils import make_dir, print_args, get_time
from utils import LOG_DIR, DATA_DIR, CKPT_DIR
from utils import LABELS, eval_metric
from utils import ID2LABEL, ID2LABEL_ABSA


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
    if not args.use_lr_scheduler:
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
        ttl_cpc_loss / cpc_batch_count,
        ttl_absa_loss / absa_batch_count
    )

    return msg


def move_batch_to_gpu(batch):
    """get all entries of tensors in the batch by `BATCH_TENSORS`
    and move them to GPU"""
    for field in BATCH_TENSORS:
        batch[field] = batch[field].cuda()
    return batch


def compute_metrics(on_gpu, pred_list, gt_list):
    """
    computing the performance using two lists of vectors
    Args:
        on_gpu - whether the input vectors are on gpus
        pred_list - list of torch tensors of predictions
        gt_list - lits of torch tensor of ground truths
    Return:
        metric_dict = 
            {
                0: f1(cls 0),
                1: f1(cls 1),
                2: f1(cls 2),
                "metric": f1(micro)
            }
    """
    if on_gpu:
        pred_list = [x.cpu() for x in pred_list]
        gt_list = [x.cpu() for x in gt_list]
    
    pred_vec = np.concatenate([x.detach().numpy() for x in pred_list])
    gt_vec = np.concatenate([x.detach().numpy() for x in gt_list])
    return eval_metric(y_true=gt_vec, y_pred=pred_vec, labels=LABELS)


def compose_metric_perf_msg(metric_dict):
    perf_msg = + " ".join(
        [f"{ID2LABEL[i]}-{metric_dict[i]:.6f};" for i in LABELS]) \
        + f" micro-{metric_dict['micro']}"
    return perf_msg
    


def train(args, use_gpu, model, dataloader):

    logging.info(f"[info] start training ID:[{args.experimentID}]")
    print(f"[info] start training ID:[{args.experimentID}]")

    # create optimizer(s) and schedulers
    # configs of optimizers and schedulers depend on `use_single_optimizer`
    # and `use_lr_scheduler`.
    cpc_optim, cpc_sched = get_optimizer_and_scheduler(args, CPC, model)
    if not args.use_single_optimizer:
        absa_optim, absa_sched = get_optimizer_and_scheduler(args, ABSA, model)

    # create loss function
    criterion = nn.CrossEntropyLoss()

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
            if use_gpu:
                batch = move_batch_to_gpu(batch)
            
            # get prediction and target
            pred = model(batch)
            target = torch.tensor([x.get_label_id() for x in batch['instances']])

            # choose optimizer and zero grad
            if not args.use_single_optimizer and task == ABSA:
                optim = absa_optim
            else:
                optim = cpc_optim
            optim.zero_grad()

            # compute loss, compute derivation, optimize
            loss = criterion(pred, target)
            loss.backward()
            optim.step()

            # accumulate loss
            if task == CPC:
                cpc_batch_count += 1
                total_cpc_loss += loss.item()
                # collect prediction results and ground truths
                predictions.append(pred)
                groundtruths.append(target)

                if cpc_batch_count % args.log_batch_num:
                    msg = compose_msg(CPC, ep, cpc_batch_count, 
                        total_iter_num_per_epoch, loss.item(), 
                        total_cpc_loss, cpc_batch_count,
                        total_absa_loss, absa_batch_count)
            else:
                absa_batch_count += 1
                total_absa_loss += loss.item()
                if absa_batch_count % args.absa_log_batch_num:
                    msg = compose_msg(ABSA, ep, cpc_batch_count, 
                        total_iter_num_per_epoch, loss.item(), 
                        total_cpc_loss, cpc_batch_count,
                        total_absa_loss, absa_batch_count)
            
            logging.info("[Perf][Iter] " + msg)
            print(f"{get_time()} [Perf][Iter] {msg}")

            
        # log training loss after each epoch
        msg = compose_msg(CPC, ep, 0, 0, 0, total_cpc_loss, cpc_batch_count,
            total_absa_loss, absa_batch_count)
        logging.info("[Perf][Epoch] " + msg)
        print(f"{get_time{}} [Perf][Epoch] {msg}")

        # compute and log training performance after each epoch
        metric_dict = compute_metrics(use_gpu, predictions, groundtruths)
        perf_msg = compose_metric_perf_msg(metric_dict)
        logging.info(f"[Perf][Train][CPC][Epoch]{ep} " + perf_msg)
        print(f"{get_time()} [Perf][Train][CPC][Epoch]{ep} " + perf_msg)

        # averaging loss for epoch
        if args.use_lr_scheduler:
            cpc_sched.step()
            if absa_sched:
                absa_sched.step()

        # run validation
        if ep % args.eval_per_ep and ep >= args.eval_after_epnum - 1:
            prec, recall, f1 = evaluate(model, for_test=False,
                dataloader=dataloader.get_batch_testval(False), 
                restore_model_path=None)
            msg = compose_msg()
            # TODO: change compose msg to add pred/recall/f1
            logging.info("[val] "+msg)
            print(f"{get_time()} [Perf][val] {msg}")

        # save model
        if args.save_model and not ep % args.save_per_ep \
            and ep >= args.save_after_epnum - 1:
            model_name = "model_expID{}_ep{}".format(args.experimentID, ep)
            torch.save(model.state_dict(), args.save_model_path+model_name)
            logging.info(f"[save] saving model: {model_name}")
            print(f"{get_time()} [save] saving model: {model_name}")


def evaluate(model, for_test, dataloader, restore_model_path, use_gpu):
    model.eval()
    predictions, groundtruths = [], []
    use_gpu = use_gpu and torch.cuda.is_available()

    task = "Test" if for_test else "Val"

    if restore_model_path:
        logging.info(f"[Eval] loading model from {restore_model_path}")
        print(f"[Eval] loading model from {restore_model_path}")
        model.load_state_dict(torch.load(restore_model_path))

    with torch.no_grad():
        for i, eval_batch in enumerate(dataloader):
            if use_gpu:
                batch = move_batch_to_gpu(batch)
            eval_pred = model(batch)
            eval_groundtruth =torch.tensor(
                [x.get_label_id() for x in batch['instances']])
            if use_gpu:
                eval_pred = eval_pred.cpu()
                eval_groundtruth = eval_groundtruth.cpu()
            predictions.append(eval_pred.detach().numpy())
            groundtruths.append(eval_groundtruth.detach().numpy())
        predictions = np.concatenate(predictions)
        groundtruths = np.concatenate(groundtruths)
        metric_dict = compute_metrics(use_gpu, predictions, groundtruths)
        perf_msg = compose_metric_perf_msg(metric_dict)
        logging.info(f"[Perf][CPC][{task}] " + perf_msg)
        print(f"{get_time()} [Perf][CPC][{task}] " + perf_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # environment config
    parser.add_argument("--experimentID", type=str, required=True, help="The ID of the experiments")
    parser.add_argument("--task", type=str, required=True, help="Train/Test?")
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--shuffle", action="store_true", default=False,
        help="Whether to shuffle data before a new epoch")

    # scheduler
    parser.add_argument("--use_lr_scheduler", action="store_true", default=False,
        help="Use this flag to turn on learning rating scheduler in model")
    parser.add_argument("--scheduler_stepsize", type=int, default=5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.8)

    # input embedding
    parser.add_argument("--input_emb", type=str, default="bert_fixed", 
        help="Select from `fix`, `ft`, and `glove`")
    parser.add_argument("--bert_version", type=str, default="bert-base-uncased")
    parser.add_argument("--glove_dim", type=int, default=100)

    # training config
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--absa_lr", type=float, default=0.005)
    parser.add_argument("--reg_weight", type=float, default=0.0001)
    parser.add_argument("--num_ep", type=int, default=10)
    parser.add_argument("--use_single_optimizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_ratio", type=str, default="1:1", 
        help="Ratio of batch numbers for CPC and ABSA. Default: 1:1.")
    parser.add_argument("--data_augmentation", action="store_true", default=False,
        help="Whether to conduct data augmentation for training set. Default=False.")

    # logging 
    parser.add_argument("--log_batch_num", type=int, default=100)
    parser.add_argument("--absa_log_batch_num", type=int, default=100)
    
    # validation
    parser.add_argument("--eval_per_ep", type=int, default="100",
        help="Run validation every this number of epochs. Default=100 (No val)")
    parser.add_argument("--eval_after_num", type=int, default="100",
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
        use_gpu = True
        assert torch.cuda.device_count() > args.gpu_id
        torch.cuda.set_device("cuda:"+str(args.gpu_id))
        msg = "[cuda] with {} gpus, using cuda:{}".format(
            torch.cuda.device_count(), args.gpu_id)
    else:
        use_gpu = False
        msg = "[cuda] no gpus, using cpu"


    logging.info(msg)
    print("{} {}".format(get_time(), msg))

    # set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    dataloader = DataLoader(args)

    # move model to cuda device
    # TODO: check on models
    model = SaeccModel(args)
    if use_gpu:
        model = model.cuda()

    if args.task == "train":
        train(args, use_gpu, model, dataloader)
    elif args.task == "test":
        evaluate(model, dataloader=dataloader.get_batch_testval(for_test=True),
            restore_model_path=args.load_model_path)
    else:
        raise ValueError("args.task can only be train or test")