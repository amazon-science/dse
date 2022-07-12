import os
import csv
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset

class PairSamples(Dataset):
    def __init__(self, train_x1, train_x2, pairsimi):
        assert len(pairsimi) == len(train_x1) == len(train_x2)
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.pairsimi = pairsimi
        
        
    def __len__(self):
        return len(self.pairsimi)

    def __getitem__(self, idx):
        return {'text1': self.train_x1[idx], 'text2': self.train_x2[idx], 'pairsimi': self.pairsimi[idx]}


'''
Assumed data format:

sentence1, sentence2

'''
def pair_loader_csv(args):
    delimiter = "," if args.dataname.endswith(".csv") else "\t"
    with open(os.path.join(args.datapath, args.dataname), "r") as f:
        train_data = list(csv.reader(f, delimiter=delimiter))
    train_text1 = [d[0] for d in train_data]
    train_text2 = [d[1] for d in train_data]
    pairsimi = [1 for _ in train_data]

    train_dataset = PairSamples(train_text1, train_text2, pairsimi)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader


'''
Expect a txt file where each line contains a single sentence/paragraph.
'''
def pair_loader_txt(args):
    with open(os.path.join(args.datapath, args.dataname), "r") as f:
        texts = f.readlines()
        texts = [t.strip("\n") for t in texts]

    train_text1 = texts
    train_text2 = texts
    pairsimi = [1] * len(texts)

    train_dataset = PairSamples(train_text1, train_text2, pairsimi)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader