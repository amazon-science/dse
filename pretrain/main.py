import os 
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import argparse
import torch
import torch.nn as nn

from models.Transformers import PSCBert, PSCRoberta, PSCDistilBERT
from training import PSCTrainer
from dataloader.dataloader import pair_loader_csv, pair_loader_txt
from utils.utils import set_global_random_seed, setup_path
from utils.optimizer import get_optimizer, get_bert_config_tokenizer, MODEL_CLASS
import subprocess
    
def run(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(device_id))

    '''
    We assume paired training data (e.g., NLI data) is always saved in csv/tsv format,
    and single training data (e.g., wiki) is always saved in txt format.
    '''
    if args.dataname.endswith(".csv") or args.dataname.endswith(".tsv"):
        train_loader = pair_loader_csv(args)
    elif args.dataname.endswith(".txt"):
        train_loader = pair_loader_txt(args)
    else:
        return ValueError()
    
    # model & optimizer
    config, tokenizer = get_bert_config_tokenizer(args.bert)
    if 'roberta' in args.bert:
        model = PSCRoberta.from_pretrained(MODEL_CLASS[args.bert], feat_dim=args.feat_dim)
    elif 'distilbert' in args.bert:
        model = PSCDistilBERT.from_pretrained(MODEL_CLASS[args.bert], feat_dim=args.feat_dim)
    else:
        model = PSCBert.from_pretrained(MODEL_CLASS[args.bert], feat_dim=args.feat_dim)

    optimizer = get_optimizer(model, args)
    
    model = nn.DataParallel(model)
    model.to(device)
    
    # set up the trainer
    trainer = PSCTrainer(model, tokenizer, optimizer, train_loader, args)
    trainer.train()
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--resdir', type=str, default='./results')
    parser.add_argument('--logging_step', type=int, default=250, help="")
    # Dataset
    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--dataname', type=str, default='tod_single_pos_3.tsv', help="")
    # Training parameters
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-06, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_iter', type=int, default=100000000)
    # Contrastive learning
    parser.add_argument('--mode', type=str, default='contrastive', help="")
    parser.add_argument('--bert', type=str, default='distilbert', help="")
    parser.add_argument('--contrast_type', type=str, default="HardNeg")
    parser.add_argument('--feat_dim', type=int, default=128, help="dimension of the projected features for instance discrimination loss")
    parser.add_argument('--decay_rate', type=float, default=1, help="the decay rate when modeling multi-turn dialogue")
    parser.add_argument('--num_turn', type=int, default=1, help="number of previous turn used in model training and response selection")
    parser.add_argument('--temperature', type=float, default=0.05, help="temperature required by contrastive loss")
    parser.add_argument('--save_model_every_epoch', action='store_true', default=True, help="Whether to save model at every epoch")

    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    run(args)




    


