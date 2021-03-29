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

from model import SaeccModel
from dataloader import DataLoader
from utils import make_dir, print_args, get_time
from utils import LOG_DIR, DATA_DIR, CKPT_DIR


def train(args, model, dataloader):
    """TODO"""
    pass


def evaluate(model, test_dl, restore_model_path):
    """TODO"""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimentID", type=str, required=True, help="The ID of the experiments")
    parser.add_argument("--task", type=str, required=True, help="Train/Test?")
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument("--shuffle", action="store_true", default=False,
        help="Whether to shuffle data before a new epoch")

    parser.add_argument("--load_model", action="store_true", default=False,
        help="Whether to resume training from an existing ckpt")
    parser.add_argument("--load_model_path", type=str, help="Path to load existing model")

    # input embedding setting: bert (fixed/fine-tune) and glove
    parser.add_argument("--input_emb", type=str, default="bert_fixed", 
        help="Select from `fix`, `ft`, and `glove`")
    parser.add_argument("--bert_version", type=str, default="bert-base-uncased")
    parser.add_argument("--glove_dim", type=int, default=100)

    # training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_ratio", type=str, default="1:1", 
        help="Ratio of batch numbers for CPC and ABSA. Default: 1:1.")
    parser.add_argument("--data_augmentation", action="store_true", default=False,
        help="Whether to conduct data augmentation for training set. Default=False.")

    
    # save model configeration
    parser.add_argument("--save_model", action="store_true", default=False, 
        help="Whether to turn on model saving.")
    parser.add_argument("--save_epnum", type=int, default=2, help="Save model per x epochs.")
    parser.add_argument("--save_after_epnum", type=int, default=1000,
        help="Number of iterations per model saving." + 
             "Only in effect when `save_model` is turned on.")
    parser.add_argument("--save_model_path", type=str, default="./ckpt/",
        help="Path to directory to save models")

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
    model = SaeccModel(args)
    if use_gpu:
        model = model.cuda()

    if args.task == "train":
        train(args, model, dataloader)
    elif args.task == "test":
        evaluate(model, test_dl=dataloader.get_batch_iterator(for_test=True),
            restore_model_path=args.load_model_path)
    else:
        raise ValueError("args.task can only be train or test")