import os
import sys
import csv
import numpy as np

import torch
import torch.nn as nn
from utils.contrastive_utils import HardConLoss
from utils.utils import statistics_log 


from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import normalize
from tqdm import tqdm







class PSCTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(PSCTrainer, self).__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.task_type = self.args.mode
        self.gstep = 0
        self.dev_objective = -1
        
        self.psc_loss = HardConLoss(temperature=self.args.temperature, contrast_type=self.args.contrast_type).cuda()
        self.classify_loss = nn.CrossEntropyLoss().cuda()
        print("\nUsing PSC_Trainer, {}\n".format(self.args.contrast_type))
        

    def get_batch_token(self, text, max_length=-1):
        if max_length == -1:
            max_length = self.args.max_length

        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_pairwise_input(self, batch):
        text1, text2, pairsimi = batch['text1'], batch['text2'], batch['pairsimi'].cuda()
        feat1 = self.get_batch_token(text1)
        feat2 = self.get_batch_token(text2)

        
        input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids.cuda(), attention_mask.cuda(), pairsimi.detach()
    


    def prepare_pairwise_input_multiturn_concatenate(self, batch):
        text1, text2, pairsimi = batch['text1'], batch['text2'], batch['pairsimi'].cuda()
        max_query_length = self.args.num_turn * self.args.max_length
        num_keeped_words = int(max_query_length*0.9)
        text1 = [" ".join(t.split()[-num_keeped_words:]) for t in text1]
        feat1 = self.get_batch_token(text1, max_length=max_query_length)
        feat2 = self.get_batch_token(text2, max_length=32)


        batch_size = feat2['input_ids'].shape[0]
        seq_length = feat2['input_ids'].shape[1]



        input_ids = torch.cat([feat1['input_ids'].reshape(batch_size, -1, seq_length), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].reshape(batch_size, -1, seq_length), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids.cuda(), attention_mask.cuda(), pairsimi.detach()



    def save_model(self, epoch, best_dev=False):
        if best_dev:
            save_dir = os.path.join(self.args.resPath, 'dev')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.module.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
        else:
            save_dir = os.path.join(self.args.resPath, str(epoch+1))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.module.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)


    def train(self):

        all_iter = self.args.epochs * len(self.train_loader)
        print('\n={}/{}=Iterations/Batches'.format(all_iter, len(self.train_loader)))

        self.model.train()
        epoch_iterator = tqdm(self.train_loader, desc="Iteration")
        for epoch in range(self.args.epochs):
            for j, batch in enumerate(epoch_iterator):
                if self.args.num_turn > 1:
                    input_ids, attention_mask, pairsimi = self.prepare_pairwise_input_multiturn_concatenate(batch)
                else:
                    input_ids, attention_mask, pairsimi = self.prepare_pairwise_input(batch)
                    
                losses = self.train_step(input_ids, attention_mask, pairsimi)


                if (self.gstep%self.args.logging_step==0) or (self.gstep==all_iter) or (self.gstep==self.args.max_iter):
                    statistics_log(self.args.tensorboard, losses=losses, global_step=self.gstep)
                        
                elif self.gstep > self.args.max_iter:
                    break
                    
                self.gstep += 1
                
            print("Finish Epoch: ", epoch)
            if self.args.save_model_every_epoch:
                self.save_model(epoch, best_dev=False)
        return None
        

    def train_step(self, input_ids, attention_mask, pairsimi, speaker_query_labels=None, speaker_response_labels=None):         
        feat1, feat2, _, _ = self.model(input_ids, attention_mask, task_type='contrastive')
        losses = self.psc_loss(feat1, feat2, pairsimi)
        loss = losses["instdisc_loss"]

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    

    


