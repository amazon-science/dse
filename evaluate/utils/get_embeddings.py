import os
import argparse
import json
import time
import pickle
from tqdm import tqdm, trange
import numpy as np
from sklearn.preprocessing import normalize

from transformers import AutoModel, AutoTokenizer

import torch
from torch.utils.data import DataLoader, SequentialSampler



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def _tokenize_multiturn_dialogue_split(raw_text, num_turn, tokenizer, max_seq_len):
    cleaned_text = [t.replace("[SEP]", "***").strip("***") for t in raw_text]
    split_text = [t.split("***") for t in cleaned_text]

    turn_count = [len(t) for t in split_text]
    print("Average number of turns: ", np.mean(turn_count))
    print("Max number of turns: ", np.max(turn_count))
    print("Minimum number of turn to cover 95%", np.percentile(turn_count, 95))

    split_text = [["[PAD]"] * max(num_turn - len(t),0) + t[-num_turn:] for t in split_text]
    print(split_text[:2])
    text = []
    for txt in split_text:
        text += txt
    text = [t.strip() for t in text]
    

    token_feat = tokenizer.batch_encode_plus(
        text, 
        max_length=max_seq_len, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )

    batch_size = len(cleaned_text)
    input_ids = token_feat['input_ids'].view(batch_size, num_turn, -1)
    attention_mask = token_feat['attention_mask'].view(batch_size, num_turn, -1)


    encodings = {"input_ids": input_ids, 
                "attention_mask": attention_mask}
    
    return encodings

def calculate_embedding(texts, model, tokenizer, device='cuda', batch_size=1000, max_length=32, verbose=False, task_type="average_embedding", num_category=0, num_turn=1):
    n_gpu = torch.cuda.device_count()

    hidden_size = model.config.hidden_size
    model.to(device)
    
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    if task_type == "dialogpt_embedding":
        batch_size = int(batch_size/20)

    if task_type == "rnn_embedding" and num_turn > 1:
        encoding = _tokenize_multiturn_dialogue_split(texts, num_turn, tokenizer, 48)
    elif task_type == "dialogpt_embedding":
        encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    else:
        encoding = tokenizer(texts, padding="longest", truncation=True, return_tensors="pt", max_length=max_length)
    emb_dataset = TextDataset(encoding)
    emb_sampler = SequentialSampler(emb_dataset)
    emb_dataloader = DataLoader(emb_dataset, batch_size=batch_size * n_gpu, sampler=emb_sampler)
    emb_iterator = tqdm(emb_dataloader, desc="Iteration") if verbose else emb_dataloader

    if task_type in ["cls_embedding", "cls_embedding_nopool", "average_embedding", "rnn_embedding", "dialogpt_embedding"]:
        results = torch.zeros([1, hidden_size], dtype=torch.float16)
    elif task_type == "classification":
        softmax = torch.nn.Softmax(dim=1)
        results = torch.zeros([1, num_category], dtype=torch.float32)
    else:
        raise ValueError("Please choose task_type from [cls_embedding, average_embedding, classification]")

    with torch.no_grad():
        model.eval()
        for inputs in emb_iterator:
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            if task_type == "dialogpt_embedding":
                attention_mask = (input_ids != 50256).long()

            if task_type in ["cls_embedding", "cls_embedding_nopool", "average_embedding", "rnn_embedding", "dialogpt_embedding"]:
                embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
                if task_type == "cls_embedding":
                    embeddings = embeddings[1].to("cpu").type_as(results)
                elif task_type == "cls_embedding_nopool":
                    embeddings = embeddings[0][:, 0, :].to("cpu").type_as(results)
                elif task_type == "average_embedding":
                    attention_mask = attention_mask.unsqueeze(-1)
                    embeddings = torch.sum(embeddings[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
                    embeddings = embeddings.to("cpu").type_as(results)
                elif task_type == "dialogpt_embedding":
                    # last_token_position = attention_mask.sum(1) - 1
                    # embeddings = embeddings[0]
                    # embeddings = embeddings[torch.arange(embeddings.shape[0]), last_token_position]
                    attention_mask = attention_mask.unsqueeze(-1)
                    embeddings = torch.sum(embeddings[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
                    embeddings = embeddings.to("cpu").type_as(results)
                elif task_type == "rnn_embedding":
                    embeddings = embeddings.to("cpu").type_as(results)
                results = torch.cat([results, embeddings])

            elif task_type == "classification":
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cur_preds = outputs.logits.to("cpu").type_as(results)
                cur_preds = softmax(cur_preds)
                results = torch.cat([results, cur_preds])

            # elif task_type == "dialogpt_embedding":
            #     print(attention_mask.shape)
            #     transformer_outputs = model(
            #         input_ids,
            #         attention_mask)[0]
            #     embeddings = transformer_outputs.mean(1)
            #     embeddings = embeddings.to("cpu").type_as(results)
            #     results = torch.cat([results, embeddings])
                
    
    return results[1:].numpy().astype(np.float32)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size of bert embedding calculation", default=0)
    parser.add_argument("--n_gpu", type=int, help="Batch size of bert embedding calculation", default=1)
    parser.add_argument("--text_dir", type=str, help="path to the original text file", default="/mnt/efs/ToD-BERT/data_tod/all_sent_cleaned.txt")
    parser.add_argument("--embedding_dir", type=str, help="path to the embedding file", default="data/embedding.npy")
    parser.add_argument("--model_dir", type=str, help="path to the model used for calculating embedding", default="TODBERT/TOD-BERT-JNT-V1")
    parser.add_argument("--average_embedding", action="store_true", help="", default=False)
    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()
    batch_size = args.batch_size if args.batch_size else 10

    root_path = '/'.join(args.index_dir.split("/")[:-1])
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    with open(args.text_dir, "r") as f:
        texts = f.readlines()
    texts = [t.strip("\n") for t in texts]
    texts = [t.split('\t')[-1] for t in texts]


    device = "cuda" 
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModel.from_pretrained(args.model_dir)

    task_type = "average_embedding" if args.average_embedding else "cls_embedding"
    ori_embedding = calculate_embedding(texts, model, tokenizer, task_type=task_type, verbose=True)


        


    

